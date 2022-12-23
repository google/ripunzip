// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod cloneable_seekable_reader;
mod http_range_reader;
mod seekable_http_reader;

use std::{
    borrow::Cow,
    fs::{File, Permissions},
    io::{ErrorKind, Read, Seek},
    num::{NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use rayon::prelude::*;
use zip::{read::ZipFile, result::ZipResult, ZipArchive};

use crate::unzip::cloneable_seekable_reader::CloneableSeekableReader;

use self::{
    cloneable_seekable_reader::HasLength,
    seekable_http_reader::{AccessPattern, SeekableHttpReader, SeekableHttpReaderEngine},
};

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

/// We create multiple parallel HTTP streams. Each HTTP stream reads this much
/// of the end file. If we set this too low, we waste parallelism. If we set
/// it too high, each HTTP stream takes too long to ramp up based on the TCP
/// slow start algorithm (or equivalents)
const DEFAULT_RANGE_PER_READER: u64 = 1024 * 1024 * 1024;

/// Options for downloading from a URI. If in doubt, use the [`Default`]
/// implementation. These parameters are typically only useful for testing
/// or fuzzing or performance optimization, though you may wish to set
/// [`readahead_limit`] to limit RAM consumption.
#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "fuzzing", derive(arbitrary::Arbitrary))]
pub struct UnzipUriOptions {
    /// Maximum RAM consumption of stored blocks of zip data that are yet
    /// to be read by the unzip algorithm.
    pub readahead_limit: Option<usize>,
    /// How many bytes to read in one go, if we're reading ahead of the
    /// current zip file request. This is almost never worth tinkering
    /// with unless you're testing pathological cases.
    pub max_block: Option<NonZeroUsize>,
    /// The maximum amount of data to read per HTTP stream.
    pub range_per_stream: Option<NonZeroU64>,
    /// The maximum number of parallel HTTP streams to have.
    pub maximum_streams: Option<NonZeroUsize>,
}

/// Options for unzipping.
pub struct UnzipOptions {
    /// The destination directory.
    pub output_directory: Option<PathBuf>,
    /// Whether to run in single-threaded mode.
    pub single_threaded: bool,
}

/// A trait of types which wish to hear progress updates on the unzip.
pub trait UnzipProgressReporter: Sync {
    /// Extraction has begun on a file.
    fn extraction_starting(&self, _display_name: &str) {}
    /// Extraction has finished on a file.
    fn extraction_finished(&self, _display_name: &str) {}
    /// The total number of compressed bytes we expect to extract.
    fn total_bytes_expected(&self, _expected: u64) {}
    /// Some bytes of a file have been decompressed. This is probably
    /// the best way to display an overall progress bar. This should eventually
    /// add up to the number you're given using [`total_bytes_expected`].
    /// The 'count' parameter is _not_ a running total - you must add up
    /// each call to this function into the running total.
    /// It's a bit unfortunate that we give compressed bytes rather than
    /// uncompressed bytes, but currently we can't calculate uncompressed
    /// bytes without downloading the whole zip file first, which rather
    /// defeats the point.
    fn bytes_extracted(&self, _count: u64) {}
}

/// A progress reporter which does nothing.
pub struct NullProgressReporter;

impl UnzipProgressReporter for NullProgressReporter {}

/// An object which can unzip a zip file, in its entirety, from a local
/// file or from a network stream. It tries to do this in parallel wherever
/// possible.
pub struct UnzipEngine<P: UnzipProgressReporter> {
    progress_reporter: P,
    options: UnzipOptions,
    zipfile: Box<dyn UnzipEngineImpl>,
    compressed_length: u64,
    directory_creator: DirectoryCreator,
}

/// The underlying engine used by the unzipper. This is different
/// for files and URIs.
trait UnzipEngineImpl {
    fn len(&self) -> usize;
    fn by_index_raw(&mut self, i: usize) -> ZipResult<ZipFile<'_>>;
    fn unzip(
        &mut self,
        single_threaded: bool,
        output_directory: &Option<PathBuf>,
        progress_reporter: &(dyn UnzipProgressReporter + Sync),
        directory_creator: &DirectoryCreator,
    ) -> Vec<anyhow::Error>;
}

/// Engine which knows how to unzip a file.
#[derive(Clone)]
struct UnzipFileEngine(ZipArchive<CloneableSeekableReader<File>>);

impl UnzipEngineImpl for UnzipFileEngine {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn by_index_raw(&mut self, i: usize) -> ZipResult<ZipFile<'_>> {
        self.0.by_index_raw(i)
    }
    fn unzip(
        &mut self,
        single_threaded: bool,
        output_directory: &Option<PathBuf>,
        progress_reporter: &(dyn UnzipProgressReporter + Sync),
        directory_creator: &DirectoryCreator,
    ) -> Vec<anyhow::Error> {
        if single_threaded {
            (0..self.len())
                .into_iter()
                .map(|i| {
                    extract_file(
                        &mut self.0,
                        i,
                        output_directory,
                        progress_reporter,
                        directory_creator,
                    )
                })
                .filter_map(Result::err)
                .collect()
        } else {
            (0..self.len())
                .into_par_iter()
                .map(|i| {
                    extract_file(
                        &mut self.0.clone(),
                        i,
                        output_directory,
                        progress_reporter,
                        directory_creator,
                    )
                })
                .filter_map(Result::err)
                .collect()
        }
    }
}

/// Engine which knows how to unzip a URI; specifically a URI fetched from
/// an HTTP server which supports `Range` requests.
#[derive(Clone)]
struct UnzipUriEngine<F: Fn()>(
    Arc<SeekableHttpReaderEngine>,
    ZipArchive<SeekableHttpReader>,
    F,
);

impl<F: Fn()> UnzipEngineImpl for UnzipUriEngine<F> {
    fn len(&self) -> usize {
        self.1.len()
    }

    fn by_index_raw(&mut self, i: usize) -> ZipResult<ZipFile<'_>> {
        self.1.by_index_raw(i)
    }

    fn unzip(
        &mut self,
        single_threaded: bool,
        output_directory: &Option<PathBuf>,
        progress_reporter: &(dyn UnzipProgressReporter + Sync),
        directory_creator: &DirectoryCreator,
    ) -> Vec<anyhow::Error> {
        self.0
            .set_expected_access_pattern(AccessPattern::SequentialIsh);
        let result = if single_threaded {
            (0..self.len())
                .into_iter()
                .map(|i| {
                    extract_file(
                        &mut self.1,
                        i,
                        output_directory,
                        progress_reporter,
                        directory_creator,
                    )
                })
                .filter_map(Result::err)
                .collect()
        } else {
            (0..self.len())
                .into_par_iter()
                .map(|i| {
                    extract_file(
                        &mut self.1.clone(),
                        i,
                        output_directory,
                        progress_reporter,
                        directory_creator,
                    )
                })
                .filter_map(Result::err)
                .collect()
        };
        let stats = self.0.get_stats();
        if stats.cache_shrinks > 0 {
            self.2()
        }
        result
    }
}

impl<P: UnzipProgressReporter> UnzipEngine<P> {
    /// Create an unzip engine which knows how to unzip a file.
    pub fn for_file(zipfile: File, options: UnzipOptions, progress_reporter: P) -> Result<Self> {
        // The following line doesn't actually seem to make any significant
        // performance difference.
        // let zipfile = BufReader::new(zipfile);
        let compressed_length = zipfile.len();
        let zipfile = CloneableSeekableReader::new(zipfile);
        Ok(Self {
            progress_reporter,
            options,
            zipfile: Box::new(UnzipFileEngine(ZipArchive::new(zipfile)?)),
            compressed_length,
            directory_creator: DirectoryCreator::default(),
        })
    }

    /// Create an unzip engine which knows how to unzip a URI. This will try
    /// to download and unzip in parallel, if the HTTP server supports
    /// the `Range` header.
    /// Parameters:
    /// - the URI
    /// - unzip options
    /// - how big a readahead buffer to create in memory.
    /// - optionally, a maximum block size to read. This is only useful in
    ///   testing and fuzzing cases where we want to simulate maximal parallelism
    ///   even with small zip files. Leave at `None` if in doubt.
    /// - how large a chunk of the file should be downloaded on each separate
    ///   HTTP stream. Leave at `None` if in doubt.
    /// - a maximum number of streams to create
    /// - a progress reporter (set of callbacks)
    /// - an additional callback to warn if performance was impaired by
    ///   rewinding the HTTP stream. (This implies the readahead buffer was
    ///   too small.)
    pub fn for_uri<F: Fn() + 'static>(
        uri: &str,
        options: UnzipOptions,
        uri_options: UnzipUriOptions,
        progress_reporter: P,
        callback_on_rewind: F,
    ) -> Result<Self> {
        let max_block = uri_options
            .max_block
            .unwrap_or_else(|| NonZeroUsize::new(DEFAULT_MAX_BLOCK).unwrap());
        let range_per_reader = uri_options
            .range_per_stream
            .unwrap_or_else(|| NonZeroU64::new(DEFAULT_RANGE_PER_READER).unwrap());
        let seekable_http_reader = SeekableHttpReaderEngine::new(
            uri.to_string(),
            uri_options.readahead_limit,
            uri_options.maximum_streams,
            AccessPattern::RandomAccess,
            max_block,
            range_per_reader,
        );
        let (compressed_length, zipfile): (u64, Box<dyn UnzipEngineImpl>) =
            match seekable_http_reader {
                Ok(seekable_http_reader) => (
                    seekable_http_reader.len(),
                    Box::new(UnzipUriEngine(
                        seekable_http_reader.clone(),
                        ZipArchive::new(seekable_http_reader.create_reader())?,
                        callback_on_rewind,
                    )),
                ),
                Err(_) => {
                    // This server probably doesn't support HTTP ranges.
                    // Let's fall back to fetching the request into a temporary
                    // file then unzipping.
                    let mut response = reqwest::blocking::get(uri)?;
                    let mut tempfile = tempfile::tempfile()?;
                    std::io::copy(&mut response, &mut tempfile)?;
                    let compressed_length = tempfile.len();
                    let zipfile = CloneableSeekableReader::new(tempfile);
                    (
                        compressed_length,
                        Box::new(UnzipFileEngine(ZipArchive::new(zipfile)?)),
                    )
                }
            };
        Ok(Self {
            progress_reporter,
            options,
            zipfile,
            compressed_length,
            directory_creator: DirectoryCreator::default(),
        })
    }

    /// The total number of files we expect to unzip.
    pub fn file_count(&self) -> usize {
        self.zipfile.len()
    }

    /// The total compressed length that we expect to retrieve over
    /// the network or from the compressed file.
    pub fn zip_length(&self) -> u64 {
        self.compressed_length
    }

    /// Perform the unzip.
    pub fn unzip(mut self) -> Result<()> {
        log::info!("Starting extract");
        self.progress_reporter
            .total_bytes_expected(self.compressed_length);
        let output_directory = &self.options.output_directory;
        let single_threaded = self.options.single_threaded;
        let errors = self.zipfile.unzip(
            single_threaded,
            output_directory,
            &self.progress_reporter,
            &self.directory_creator,
        );
        // Return the first error code, if any.
        errors.into_iter().next().map(Result::Err).unwrap_or(Ok(()))
    }
}

/// Extracts a file from a zip file, attaching diagnostics to any errors where
/// possible.
fn extract_file<T: Read + Seek>(
    myzip: &mut zip::ZipArchive<T>,
    i: usize,
    output_directory: &Option<PathBuf>,
    progress_reporter: &dyn UnzipProgressReporter,
    directory_creator: &DirectoryCreator,
) -> Result<()> {
    let file = myzip.by_index(i)?;
    let name = file
        .enclosed_name()
        .map(Path::to_string_lossy)
        .unwrap_or_else(|| Cow::Borrowed("<unprintable>"))
        .to_string();
    extract_file_inner(file, output_directory, progress_reporter, directory_creator)
        .with_context(|| format!("Failed to extract {}", name))
}

/// Extracts a file from a zip file.
fn extract_file_inner(
    mut file: ZipFile,
    output_directory: &Option<PathBuf>,
    progress_reporter: &dyn UnzipProgressReporter,
    directory_creator: &DirectoryCreator,
) -> Result<()> {
    let name = file
        .enclosed_name()
        .ok_or_else(|| std::io::Error::new(ErrorKind::Unsupported, "path not safe to extract"))?;
    let out_path = match output_directory {
        Some(output_directory) => output_directory.join(name),
        None => PathBuf::from(name),
    };
    let display_name = name.display().to_string();
    progress_reporter.extraction_starting(&display_name);
    if file.name().ends_with('/') {
        directory_creator.create_dir_all(&out_path)?;
    } else {
        if let Some(parent) = out_path.parent() {
            directory_creator.create_dir_all(parent)?;
        }
        let mut out_file = File::create(&out_path).with_context(|| "Failed to create file")?;
        std::io::copy(&mut file, &mut out_file).with_context(|| "Failed to write directory")?;
        progress_reporter.bytes_extracted(file.compressed_size());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Some(mode) = file.unix_mode() {
            std::fs::set_permissions(&out_path, Permissions::from_mode(mode))
                .with_context(|| "Failed to set permissions")?;
        }
    }
    progress_reporter.extraction_finished(&display_name);
    Ok(())
}

/// An engine used to ensure we don't conflict in creating directories
/// between threads
#[derive(Default)]
struct DirectoryCreator(Mutex<()>);

impl DirectoryCreator {
    fn create_dir_all(&self, path: &Path) -> Result<()> {
        // Fast path - avoid locking if the directory exists
        if path.exists() {
            return Ok(());
        }
        let _exclusivity = self.0.lock().unwrap();
        if path.exists() {
            return Ok(());
        }
        std::fs::create_dir_all(path).with_context(|| "Failed to create directory")
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env::{current_dir, set_current_dir},
        fs::{read_to_string, File},
        io::{Cursor, Seek, Write},
        path::Path,
    };
    use tempfile::tempdir;
    use test_log::test;
    use zip::{write::FileOptions, ZipWriter};

    use crate::{NullProgressReporter, UnzipEngine, UnzipOptions, UnzipUriOptions};
    use ripunzip_test_utils::*;

    fn create_zip_file(path: &Path) {
        let file = File::create(path).unwrap();
        create_zip(file)
    }

    fn create_zip(w: impl Write + Seek) {
        let mut zip = ZipWriter::new(w);

        zip.add_directory("test/", Default::default()).unwrap();
        let options = FileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .unix_permissions(0o755);
        zip.start_file("test/a.txt", options).unwrap();
        zip.write_all(b"Contents of A\n").unwrap();
        zip.start_file("b.txt", options).unwrap();
        zip.write_all(b"Contents of B\n").unwrap();
        zip.start_file("test/c.txt", options).unwrap();
        zip.write_all(b"Contents of C\n").unwrap();
        zip.finish().unwrap();
    }

    fn check_files_exist(path: &Path) {
        let a = path.join("test/a.txt");
        let b = path.join("b.txt");
        let c = path.join("test/c.txt");
        assert_eq!(read_to_string(a).unwrap(), "Contents of A\n");
        assert_eq!(read_to_string(b).unwrap(), "Contents of B\n");
        assert_eq!(read_to_string(c).unwrap(), "Contents of C\n");
    }

    #[test]
    #[ignore] // because the chdir changes global state
    fn test_extract_no_path() {
        let td = tempdir().unwrap();
        let zf = td.path().join("z.zip");
        create_zip_file(&zf);
        let zf = File::open(zf).unwrap();
        let old_dir = current_dir().unwrap();
        set_current_dir(td.path()).unwrap();
        let options = UnzipOptions {
            output_directory: None,
            single_threaded: false,
        };
        UnzipEngine::for_file(zf, options, NullProgressReporter)
            .unwrap()
            .unzip()
            .unwrap();
        set_current_dir(old_dir).unwrap();
        check_files_exist(td.path());
    }

    #[test]
    fn test_extract_with_path() {
        let td = tempdir().unwrap();
        let zf = td.path().join("z.zip");
        create_zip_file(&zf);
        let zf = File::open(zf).unwrap();
        let outdir = td.path().join("outdir");
        let options = UnzipOptions {
            output_directory: Some(outdir.clone()),
            single_threaded: false,
        };
        UnzipEngine::for_file(zf, options, NullProgressReporter)
            .unwrap()
            .unzip()
            .unwrap();
        check_files_exist(&outdir);
    }

    use httptest::Server;

    #[test]
    fn test_extract_from_server() {
        let td = tempdir().unwrap();
        let mut zip_data = Cursor::new(Vec::new());
        create_zip(&mut zip_data);
        let body = zip_data.into_inner();
        println!("Whole zip:");
        hexdump::hexdump(&body);

        let server = Server::run();
        set_up_server(&server, body, ServerType::Ranges);

        let outdir = td.path().join("outdir");
        let options = UnzipOptions {
            output_directory: Some(outdir.clone()),
            single_threaded: false,
        };
        UnzipEngine::for_uri(
            &server.url("/foo").to_string(),
            options,
            UnzipUriOptions::default(),
            NullProgressReporter,
            || {},
        )
        .unwrap()
        .unzip()
        .unwrap();
        check_files_exist(&outdir);
    }

    fn unzip_sample_zip(zip_params: ZipParams, server_type: ServerType) {
        let td = tempdir().unwrap();
        let zip_data = ripunzip_test_utils::get_sample_zip(&zip_params);

        let server = Server::run();
        set_up_server(&server, zip_data, server_type);

        let outdir = td.path().join("outdir");
        let options = UnzipOptions {
            output_directory: Some(outdir),
            single_threaded: false,
        };
        UnzipEngine::for_uri(
            &server.url("/foo").to_string(),
            options,
            UnzipUriOptions::default(),
            NullProgressReporter,
            || {},
        )
        .unwrap()
        .unzip()
        .unwrap();
    }

    #[test]
    fn test_extract_biggish_zip_from_ranges_server() {
        unzip_sample_zip(
            ZipParams::new(FileSizes::Variable, 15, zip::CompressionMethod::Deflated),
            ServerType::Ranges,
        )
    }

    #[test]
    fn test_small_zip_from_ranges_server() {
        unzip_sample_zip(
            ZipParams::new(FileSizes::Variable, 3, zip::CompressionMethod::Deflated),
            ServerType::Ranges,
        )
    }

    #[test]
    fn test_small_zip_from_no_range_server() {
        unzip_sample_zip(
            ZipParams::new(FileSizes::Variable, 3, zip::CompressionMethod::Deflated),
            ServerType::ContentLengthButNoRanges,
        )
    }

    #[test]
    fn test_small_zip_from_no_content_length_server() {
        unzip_sample_zip(
            ZipParams::new(FileSizes::Variable, 3, zip::CompressionMethod::Deflated),
            ServerType::NoContentLength,
        )
    }
}
