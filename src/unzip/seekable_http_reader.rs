// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{
    cmp::min,
    collections::BTreeMap,
    io::{BufReader, ErrorKind, Read, Seek, SeekFrom},
    ops::Range,
    sync::{Arc, Condvar, Mutex},
};

use ranges::Ranges;
use reqwest::blocking::Response;
use thiserror::Error;

use super::http_range_reader::{self, RangeFetcher};

/// This is how much we read from the underlying HTTP stream in a given thread,
/// before signalling other threads that they may wish to continue with their
/// CPU-bound unzipping. Empirically determined.
/// 128KB = 172ms
/// 512KB = 187ms
/// 1024KB = 152ms
/// 2048KB = 170ms
/// If we set this too high, we starve multiple threads - they can't start
/// acting on the data to unzip their files until the read is complete. If we
/// set this too low, the cache structure (a `BTreeMap`) becomes dominant in
/// CPU usage.
const DEFAULT_MAX_BLOCK: usize = 1024 * 1024;

/// If we're going to skip over this much data in the underlying stream,
/// discard the stream and start further ahead. This is a large number
/// because it's expensive to create new HTTPS streams, and we also can't
/// be 100% sure we won't need bytes during this gap which might cause
/// an expensive rewind.
const DEFAULT_SKIP_AHEAD_THRESHOLD: u64 = 2 * 1024 * 1024; // 2MB

/// A hint to the [`SeekableHttpReaderEngine`] about the expected access pattern.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum AccessPattern {
    /// We expect accesses all over the file.
    RandomAccess,
    /// We expect accesses starting from the beginning and moving to the end,
    /// though there might be some jumping around if multiple threads are
    /// reading from roughly the same area of the file.
    SequentialIsh,
}

impl Default for AccessPattern {
    fn default() -> Self {
        Self::RandomAccess
    }
}

/// Errors that may be returned by a [`SeekableHttpReaderEngine` or `SeekableHttpReader`].
#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error(
        "This HTTP resource did not advertise that it accepts ranges via the Accept-Ranges header"
    )]
    AcceptRangesNotSupported,
    #[error(transparent)]
    RangeFetcherError(http_range_reader::Error),
}

/// Some data that we've read from the network, but not yet returned to the
/// caller.
struct CacheCell {
    data: Vec<u8>,
    bytes_read: Ranges<usize>,
}

impl CacheCell {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            bytes_read: Ranges::new(),
        }
    }

    fn read(&mut self, range: Range<usize>) -> &[u8] {
        let new_range = self.bytes_read.clone().union(Ranges::from(range.clone()));
        self.bytes_read = new_range;
        &self.data[range]
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn entirely_consumed(&self) -> bool {
        let data_left_to_read =
            Ranges::from(0..self.data.len()).difference(self.bytes_read.clone());
        data_left_to_read.is_empty()
    }
}

/// Internal state of the [`SeekableHttpReaderEngine`], in a separate struct
/// because access is protected by a mutex.
#[derive(Default)]
struct State {
    /// The expected pattern of seeks and reads; a hint from the user.
    access_pattern: AccessPattern,
    /// Maximum size of the "cache"
    readahead_limit: Option<usize>,
    /// Current size of the cache
    current_size: usize,
    /// The readahead "cache", which is not really a cache in the strict sense,
    /// but is any data that we've already read from the underlying stream
    /// that is yet to be read by any reader.
    /// This exists because we assume we'll get accesses in any random order,
    /// and yet we don't want to create a new HTTP stream each time we need
    /// to rewind a bit. Therefore if we fast-forward, we store any data that
    /// we skipped over, in order to service any subsequent requests for those
    /// positions.
    cache: BTreeMap<u64, CacheCell>,
    /// We expect to skip some range of the read.
    expect_skip_ahead: bool,
    /// Threshold for fast forwards when we'd expect to skip some data
    /// and decide to create a new stream.
    skip_ahead_threshold: u64,
    /// How much to read from the underlying stream each time.
    max_block: usize,
    /// Some statistics about how we're doing.
    stats: SeekableHttpReaderStatistics,
    /// Facilities to read from the underlying HTTP stream(s).
    /// If this is present, a thread may start a read - it must `take`
    /// this. If it's absent, some other thread is doing a read, and
    /// you may not.
    reader: Option<Box<ReadingMaterials>>,
    /// Some problem was encountered creating a reader.
    /// All threads should abandon hope.
    read_failed_somewhere: bool,
}

impl State {
    fn new(
        readahead_limit: Option<usize>,
        access_pattern: AccessPattern,
        skip_ahead_threshold: u64,
        max_block: usize,
        reader: Box<ReadingMaterials>,
    ) -> Self {
        // Grow the readahead limit if it's less than block size, because we
        // must always store one block in order to service the most recent read.
        let readahead_limit = match readahead_limit {
            Some(readahead_limit) if readahead_limit > max_block => Some(readahead_limit),
            Some(_) => Some(max_block),
            _ => None,
        };
        Self {
            readahead_limit,
            access_pattern,
            skip_ahead_threshold,
            max_block,
            reader: Some(reader),
            ..Default::default()
        }
    }

    /// Insert a block into our readahead cache.
    fn insert(&mut self, pos: u64, block: Vec<u8>) {
        log::debug!(
            "Inserting into cache, block is 0x{:x}-0x{:x}",
            pos,
            pos + block.len() as u64
        );
        let extra_size = block.len();
        self.cache.insert(pos, CacheCell::new(block));
        self.current_size += extra_size;
        if let Some(readahead_limit) = self.readahead_limit {
            // Shrink
            while self.current_size > readahead_limit {
                self.stats.cache_shrinks += 1;
                let first_block = self.cache.iter().next().map(|(pos, _)| pos).cloned();
                if let Some(pos) = first_block {
                    let block = self.cache.remove(&pos).unwrap();
                    self.current_size -= block.len();
                }
            }
        }
    }

    /// Read from the readahead cache, if we can.
    /// If '`discard_read_data` is true, we assume that all data
    /// will be consumed exactly once, so we discard the data that has been read.
    /// Sometimes we'll have blocks of data where we only want to read part of it,
    /// so then we will split the block and merely retain the bits that are
    /// not yet read by the readers.
    fn read_from_cache(&mut self, pos: u64, buf: &mut [u8]) -> Option<usize> {
        let discard_read_data = matches!(self.access_pattern, AccessPattern::SequentialIsh);
        let mut block_to_discard = None;
        let mut return_value = None;
        for (possible_block_start, block) in self
            .cache
            .range_mut(pos - min(pos, self.max_block as u64)..=pos)
        {
            let block_offset = pos as usize - *possible_block_start as usize;
            let block_len = block.len();
            if block_offset >= block_len {
                // This block is indeed before the read we want to do,
                // but doesn't extend as far as the starting point of our read.
                continue;
            }
            // OK, we've found a block which overlaps with the read that we
            // want to do.

            let block_len = block.len();
            let block_offset = pos as usize - *possible_block_start as usize;
            let to_read = min(buf.len(), block_len - block_offset);
            buf[..to_read].copy_from_slice(block.read(block_offset..to_read + block_offset));
            self.stats.cache_hits += 1;
            if discard_read_data && block.entirely_consumed() {
                // Discard this block, but outside this loop
                block_to_discard = Some(*possible_block_start);
                self.current_size -= block.len();
            }
            return_value = Some(to_read);
            break;
        }
        if let Some(block_to_discard) = block_to_discard {
            self.cache.remove(&block_to_discard);
        }
        return_value
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache")
            .field("max_size", &self.readahead_limit)
            .field("current_size", &self.current_size)
            .finish()
    }
}

/// Items related to reading from the underlying HTTP streams. This is
/// in a separate struct because it's protected by a mutex.
struct ReadingMaterials {
    range_fetcher: RangeFetcher,
    reader: Option<(BufReader<Response>, u64)>, // second item in tuple is current reader pos
}

/// A type which can produce objects that can be [`Read`] and [`Seek`] even
/// though they're accessing remote HTTP resources. This object in itself doesn't
/// support those traits, but its [`create_reader`] method can be used to emit
/// objects that do. This object can only be used to access HTTP resources which
/// support the `Range` header - an error will be reported on construction
/// of this object if such ranges are not supported by the remote server.
pub(crate) struct SeekableHttpReaderEngine {
    /// Total stream length
    len: u64,
    /// Overall state of this object, mostly related to the readahead cache
    /// of blocks we already read, but also with the all-important boolean
    /// stating whether any thread is already reading on the underlying stream.
    state: Mutex<State>,
    /// Condition variable to indicate that there's a new block in the
    /// readahead cache and all other threads should consider if their read
    /// request can be serviced.
    read_completed: Condvar,
}

/// Some results about the success (or otherwise) of this reader.
#[derive(Default, Debug, Clone)]
pub(crate) struct SeekableHttpReaderStatistics {
    /// The number of times we had to create an HTTP(S) stream.
    pub(crate) num_http_streams: usize,
    /// Number of times we found the read that we wanted in the cache
    /// of previous reads.
    pub(crate) cache_hits: usize,
    /// Number of times we had to actually do a read on the underlying stream.
    pub(crate) cache_misses: usize,
    /// Number of times we had to discard data from the cache because it
    /// was too big.
    pub(crate) cache_shrinks: usize,
}

impl SeekableHttpReaderEngine {
    /// Create a new seekable HTTP reader engine for this URI. This constructor
    /// will query the server to discover whether it supports HTTP ranges;
    /// if not, an error will be returned.
    pub(crate) fn new(
        uri: String,
        readahead_limit: Option<usize>,
        access_pattern: AccessPattern,
    ) -> Result<Arc<Self>, Error> {
        Self::with_configuration(
            uri,
            readahead_limit,
            access_pattern,
            DEFAULT_SKIP_AHEAD_THRESHOLD,
            DEFAULT_MAX_BLOCK,
        )
    }

    /// Constructor with a specific configuration, used for testing.
    fn with_configuration(
        uri: String,
        readahead_limit: Option<usize>,
        access_pattern: AccessPattern,
        skip_ahead_threshold: u64,
        max_block: usize,
    ) -> Result<Arc<Self>, Error> {
        let range_fetcher = RangeFetcher::new(uri).map_err(Error::RangeFetcherError)?;
        if !range_fetcher.accepts_ranges() {
            return Err(Error::AcceptRangesNotSupported);
        }
        let len = range_fetcher.len();
        Ok(Arc::new(Self {
            len,
            state: Mutex::new(State::new(
                readahead_limit,
                access_pattern,
                skip_ahead_threshold,
                max_block,
                Box::new(ReadingMaterials {
                    range_fetcher,
                    reader: None,
                }),
            )),
            read_completed: Condvar::new(),
        }))
    }

    /// Create an object which can be used to read from this HTTP location
    /// in a seekable fashion.
    pub(crate) fn create_reader(self: Arc<Self>) -> SeekableHttpReader {
        SeekableHttpReader {
            engine: self,
            pos: 0u64,
        }
    }

    /// Read some data, ideally from the cache of pre-read blocks, but
    /// otherwise from the underlying HTTP stream.
    fn read(&self, buf: &mut [u8], pos: u64) -> std::io::Result<usize> {
        // There is some mutex delicacy here. Goals are:
        // a) Allow exactly one thread to be reading on the underlying HTTP stream;
        // b) Allow other threads to query the cache of already-read blocks
        //    without blocking on ongoing reads on the stream.
        // We have a mutex which guards the overall state. Within that state
        // is a separate data structure which is used for underlying reads.
        // That struct is moved into and out of the state in order to indicate
        // whether a thread has a read in progress.
        // Specifically:
        // Claim STATE mutex
        // Is there block in cache?
        // - If yes, release STATE mutex, and return
        // - If no, check if read in progress
        //   Is there read in progress?
        //   - If yes, release STATE mutex, WAIT on condvar atomically
        //     check cache again
        //   - If no:
        //     set read in progress by taking reading materials out of state
        //     release STATE mutex
        //     perform read
        //     claim STATE mutex
        //     insert results
        //     reinsert reading materials back into the state
        //     release STATE mutex
        //     NOTIFYALL on condvar
        log::debug!("Read: requested position 0x{:x}.", pos);

        if pos == self.len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "read beyond end of stream",
            ));
        }

        // Claim CACHE mutex
        let mut state = self.state.lock().unwrap();
        // Is there block in cache?
        // - If yes, release CACHE mutex, and return
        if let Some(bytes_read_from_cache) = state.read_from_cache(pos, buf) {
            log::debug!("Immediate cache success");
            return Ok(bytes_read_from_cache);
        }
        // - If no, check if read in progress
        let mut reading_stuff = state.reader.take();
        //   Is there read in progress?
        while reading_stuff.is_none() {
            //   - If yes, release CACHE mutex, WAIT on condvar atomically
            state = self.read_completed.wait(state).unwrap();
            if state.read_failed_somewhere {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "another thread experienced a problem creating a reader",
                ));
            }
            //     check cache again
            if let Some(bytes_read_from_cache) = state.read_from_cache(pos, buf) {
                log::debug!("Deferred cache success");
                return Ok(bytes_read_from_cache);
            }
            reading_stuff = state.reader.take();
        }
        let reading_stuff = reading_stuff.unwrap(); // feels like there should
                                                    // be a way to do this with while let
        state.stats.cache_misses += 1;
        //   - If no:
        //     We'll start to do a read ourselves. Because we take()d the
        //     reading_stuff from the state, we now have exclusive ability to do the
        //     read ourselves.
        // If we need to read ahead,
        let expect_skip_ahead = state.expect_skip_ahead;
        state.expect_skip_ahead = false;
        let skip_ahead_threshold = state.skip_ahead_threshold;
        let max_block = state.max_block;
        //     release STATE mutex
        drop(state);
        //     perform read
        let read_result = self.perform_read_using_reader(
            buf,
            pos,
            reading_stuff,
            expect_skip_ahead,
            skip_ahead_threshold,
            max_block,
        );
        if read_result.is_err() {
            let mut state = self.state.lock().unwrap();
            state.read_failed_somewhere = true;
        }
        // 'state' has been updated to indicate either an error, or the
        // reading_materials has been repopulated, so wake up all other
        // threads to check.
        self.read_completed.notify_all();
        read_result
    }

    #[allow(clippy::comparison_chain)]
    // Read from the underlying HTTP stream
    // This is a separate function because if it errors at any point
    // we need to take cleanup action in the caller.
    fn perform_read_using_reader(
        &self,
        buf: &mut [u8],
        pos: u64,
        mut reading_stuff: Box<ReadingMaterials>,
        expect_skip_ahead: bool,
        skip_ahead_threshold: u64,
        max_block: usize,
    ) -> std::io::Result<usize> {
        // First check if we need to rewind, OR if we need to fast forward
        // and are expecting to skip over some significant data.
        if let Some((_, readerpos)) = reading_stuff.reader.as_ref() {
            if pos < *readerpos {
                log::debug!(
                    "Rewinding: New reader will be required at 0x{:x} - old reader pos was 0x{:x}",
                    pos,
                    *readerpos
                );
                reading_stuff.reader = None;
            } else if pos > *readerpos {
                let delta = pos - *readerpos;
                // Discard the existing stream and create a new one if we're skipping ahead a lot,
                // AND we *expect* to be skipping ahead a lot. Otherwise it might be random
                // seeks within an area backwards and forwards, and creating a new HTTP(S) stream
                // is wasteful.
                if delta > skip_ahead_threshold && expect_skip_ahead {
                    log::debug!("Fast forwarding expected skip: New reader will be required at 0x{:x} - old reader pos was 0x{:x}",
                        pos,
                        *readerpos
                    );
                    reading_stuff.reader = None;
                }
            }
        }
        let mut reader_created = false;
        if reading_stuff.reader.is_none() {
            log::debug!("create_reader");
            reading_stuff.reader = Some((
                BufReader::new(
                    reading_stuff
                        .range_fetcher
                        .fetch_range(pos)
                        .map_err(|e| std::io::Error::new(ErrorKind::Unsupported, e.to_string()))?,
                ),
                pos,
            ));
            reader_created = true;
        };

        let (reader, reader_pos) = reading_stuff.reader.as_mut().unwrap();
        if pos > *reader_pos {
            log::debug!(
                "Read: reading ahead from 0x{:x} to 0x{:x} without skipping",
                *reader_pos,
                pos
            );
        }
        while pos >= *reader_pos {
            // Fast forward beyond the desired position, recording any reads in the cache
            // for later.
            let to_read = min(max_block, self.len as usize - *reader_pos as usize);
            let mut new_block = vec![0u8; to_read];
            reader.read_exact(&mut new_block)?;
            //     claim STATE mutex
            let mut state = self.state.lock().unwrap();
            state.insert(*reader_pos, new_block);
            // Tell any waiting threads they should re-check the cache
            self.read_completed.notify_all();
            *reader_pos += to_read as u64;
        }
        // Because the above condition is >=, and because we know the request was not
        // to read at the very end of the file, we know we now have some data in the
        // cache which can satisfy the request.
        //     claim STATE mutex
        let mut state = self.state.lock().unwrap();
        let bytes_read = state
            .read_from_cache(pos, buf)
            .expect("Cache still couldn't satisfy request event after reading beyond read pos");
        log::debug!("Cache success after read");
        if reader_created {
            state.stats.num_http_streams += 1;
        }
        //     return the underlying reader to the state so that some other
        //     thread can use it
        state.reader = Some(reading_stuff);
        //     release STATE mutex
        Ok(bytes_read)
    }

    /// The total length of the underlying resource.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Update the expected access pattern. You must not call this when
    /// any threads might be reading from any [`SeekableHttpReader`] created
    /// by this engine; that may panic.
    pub(crate) fn set_expected_access_pattern(&self, access_pattern: AccessPattern) {
        let mut state = self.state.lock().unwrap();
        let old_access_pattern = state.access_pattern;
        if old_access_pattern == access_pattern {
            return;
        }
        log::debug!(
            "Changing access pattern - current stats are {:?}",
            state.stats
        );
        if matches!(access_pattern, AccessPattern::SequentialIsh) {
            // If we're switching to a sequential pattern, recreate
            // the reader at position zero.
            log::debug!("create_reader_at_zero");
            {
                let reading_materials = state.reader.as_mut().expect(
                    "Must not call set_expected_access_pattern while a read is in progress",
                );
                let new_reader = reading_materials.range_fetcher.fetch_range(0);
                if let Ok(new_reader) = new_reader {
                    reading_materials.reader = Some((BufReader::new(new_reader), 0));
                }
            }
            state.stats.num_http_streams += 1;
        }
        state.access_pattern = access_pattern;
    }

    /// Call this if we're going to skip over some part of the zip.
    pub(crate) fn read_skip_expected(&self) {
        let mut state = self.state.lock().unwrap();
        state.expect_skip_ahead = true;
    }

    /// Return some statistics about the success (or otherwise) of this stream.
    pub(crate) fn get_stats(&self) -> SeekableHttpReaderStatistics {
        self.state.lock().unwrap().stats.clone()
    }
}

impl Drop for SeekableHttpReaderEngine {
    fn drop(&mut self) {
        log::debug!("Dropping: stats are {:?}", self.state.lock().unwrap().stats)
    }
}

/// A [`Read`] which is also [`Seek`] to read from arbitrary places on an
/// HTTP stream. Cheap to clone. Create using [`SeekableHttpReader::create_reader`].
#[derive(Clone)]
pub(crate) struct SeekableHttpReader {
    engine: Arc<SeekableHttpReaderEngine>,
    pos: u64,
}

impl Seek for SeekableHttpReader {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.pos = match pos {
            SeekFrom::Start(pos) => pos,
            SeekFrom::End(pos) => {
                let positive_pos: u64 = (-pos).try_into().map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::Unsupported, "Seeked beyond end")
                })?;
                self.engine
                    .len()
                    .checked_sub(positive_pos)
                    .ok_or(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        "Rewind too far",
                    ))?
            }
            SeekFrom::Current(offset_from_pos) => {
                let offset_from_pos_u64: Result<u64, _> = offset_from_pos.try_into();
                match offset_from_pos_u64 {
                    Ok(positive_offset) => self.pos + positive_offset,
                    Err(_) => {
                        let negative_offset = -offset_from_pos as u64;
                        self.pos
                            .checked_sub(negative_offset)
                            .ok_or(std::io::Error::new(
                                std::io::ErrorKind::Unsupported,
                                "Rewind too far",
                            ))?
                    }
                }
            }
        };
        Ok(self.pos)
    }
}

impl Read for SeekableHttpReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let bytes_read = self.engine.read(buf, self.pos)?;
        self.pos += bytes_read as u64;
        Ok(bytes_read)
    }
}

#[cfg(test)]
mod tests {
    use ripunzip_test_utils::{ExpectedRange, RangeAwareResponse, RangeAwareResponseType};
    use std::io::{Read, Seek, SeekFrom};
    use test_log::test;

    use httptest::{matchers::*, Expectation, Server};

    use crate::unzip::seekable_http_reader::DEFAULT_MAX_BLOCK;

    use super::{AccessPattern, CacheCell, SeekableHttpReaderEngine};

    #[test]
    fn test_cachecell() {
        let mut cell = CacheCell::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(cell.len(), 10);
        assert!(!cell.entirely_consumed());

        assert_eq!(cell.read(0..2), &[0, 1]);
        assert!(!cell.entirely_consumed());

        assert_eq!(cell.read(3..10), &[3, 4, 5, 6, 7, 8, 9]);
        assert!(!cell.entirely_consumed());

        // Re-read some already-read bytes - still not entirely consumed.
        assert_eq!(cell.read(0..2), &[0, 1]);
        assert!(!cell.entirely_consumed());

        // Finally read that last byte.
        assert_eq!(cell.read(1..4), &[1, 2, 3]);
        assert!(cell.entirely_consumed());
    }

    #[test]
    fn test_unlimited_readahead() {
        do_test(None, AccessPattern::SequentialIsh)
    }

    #[test]
    fn test_big_readahead() {
        const ONE_HUNDRED_MB: usize = 1024usize * 1024usize * 100usize;
        do_test(Some(ONE_HUNDRED_MB), AccessPattern::SequentialIsh)
    }

    #[test]
    fn test_small_readahead() {
        do_test(Some(4), AccessPattern::SequentialIsh)
    }

    #[test]
    fn test_random_access() {
        do_test(None, AccessPattern::RandomAccess)
    }

    fn get_head_expectation() -> Expectation {
        Expectation::matching(request::method_path("HEAD", "/foo")).respond_with(
            RangeAwareResponse::new(200, RangeAwareResponseType::LengthOnly(12)),
        )
    }

    const TEST_BODY: &[u8] = "0123456789AB".as_bytes();

    fn get_range_expectation(expected_start: u64, expected_end: u64) -> Expectation {
        Expectation::matching(request::method_path("GET", "/foo"))
            .times(1..)
            .respond_with(RangeAwareResponse::new(
                206,
                RangeAwareResponseType::Body {
                    body: TEST_BODY.into(),
                    expected_range: Some(ExpectedRange {
                        expected_start,
                        expected_end,
                    }),
                },
            ))
    }

    fn do_test(readahead_limit: Option<usize>, access_pattern: AccessPattern) {
        let mut server = Server::run();
        // Expect a HEAD request first
        server.expect(get_head_expectation());

        let seekable_http_reader_engine = SeekableHttpReaderEngine::with_configuration(
            server.url("/foo").to_string(),
            readahead_limit,
            access_pattern,
            4,
            DEFAULT_MAX_BLOCK,
        )
        .unwrap();

        let mut seekable_http_reader = seekable_http_reader_engine.create_reader();
        server.verify_and_clear();

        let mut throwaway = [0u8; 4];

        // We expect a read request for the whole file
        server.expect(get_range_expectation(0, 12));
        seekable_http_reader.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "0123");
        seekable_http_reader.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "4567");
        seekable_http_reader.stream_position().unwrap();
        seekable_http_reader.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "89AB");
        server.verify_and_clear();

        // Now rewind.
        seekable_http_reader.rewind().unwrap();
        if matches!(access_pattern, AccessPattern::SequentialIsh) {
            // If we're in sequential mode, we expect to have discarded
            // the data in the cache and should start a new read.
            server.expect(get_range_expectation(0, 12));
        }
        seekable_http_reader.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "0123");
        seekable_http_reader.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "4567");
        seekable_http_reader.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "89AB");

        if matches!(access_pattern, AccessPattern::SequentialIsh) {
            server.verify_and_clear();
        }

        // Rewind a bit... we should get a range request only from here on.
        seekable_http_reader.seek(SeekFrom::Start(4)).unwrap();
        if matches!(access_pattern, AccessPattern::SequentialIsh) {
            server.expect(get_range_expectation(4, 12));
        }
        seekable_http_reader.read_exact(&mut throwaway).unwrap();
        assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "4567");

        if matches!(access_pattern, AccessPattern::SequentialIsh) {
            server.verify_and_clear();
        }

        // Test fast forwarding behavior. There's only any point
        // in sequential mode; in random access we'll service requests
        // from the cache of what we already read.

        if matches!(access_pattern, AccessPattern::SequentialIsh) {
            seekable_http_reader.rewind().unwrap();
            server.expect(get_range_expectation(0, 12));
            seekable_http_reader.read_exact(&mut throwaway).unwrap();
            assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "0123");
            server.verify_and_clear();
            // then seek forward
            seekable_http_reader.seek(SeekFrom::Start(8)).unwrap();
            // We expect no new requests. We'll just skip over.
            seekable_http_reader.read_exact(&mut throwaway).unwrap();
            assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "89AB");
            server.verify_and_clear();

            // Test fast forwarding far enough that we skip over stuff
            // enough that we might discard the HTTP stream and start a new one.
            // Create a new server where we read 4 bytes at a time from the
            // underlying stream

            server.expect(get_head_expectation());
            let seekable_http_reader_engine = SeekableHttpReaderEngine::with_configuration(
                server.url("/foo").to_string(),
                readahead_limit,
                access_pattern,
                4,
                4,
            )
            .unwrap();

            let mut seekable_http_reader = seekable_http_reader_engine.clone().create_reader();

            seekable_http_reader.rewind().unwrap();
            server.expect(get_range_expectation(0, 12));
            seekable_http_reader.read_exact(&mut throwaway).unwrap();
            assert_eq!(std::str::from_utf8(&throwaway).unwrap(), "0123");
            server.verify_and_clear();
            seekable_http_reader_engine.read_skip_expected();
            seekable_http_reader.seek(SeekFrom::Start(10)).unwrap();
            server.expect(get_range_expectation(10, 12));
            seekable_http_reader
                .read_exact(&mut throwaway[0..2])
                .unwrap();
            assert_eq!(std::str::from_utf8(&throwaway[0..2]).unwrap(), "AB");
            server.verify_and_clear();
        }
    }

    // It would be highly desirable to enhance these tests with:
    // * tests of what happens if the server refuses a connection at any
    //   point.
    // * tests of what happens if the server closes a connection part way
    //   through
    // * multi-threaded tests
}
