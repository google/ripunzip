// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Testing utilities for `ripunzip`, primarily concerned with making
//! available an HTTP server which knows how to respond with 206 codes
//! for specific ranges of data.

use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::Display,
    io::{Cursor, Seek, Write},
    sync::Mutex,
};

use arbitrary::Arbitrary;
use http::{StatusCode, Version};
use httptest::{responders::*, Expectation, Server};
use once_cell::sync::Lazy;
use regex::Regex;
use strum::IntoEnumIterator;
use zip::{write::FileOptions, ZipWriter};

/// How to respond to a range-aware request.
pub enum RangeAwareResponseType {
    LengthOnly(usize),
    Body {
        body: hyper::body::Bytes,
        expected_range: Option<ExpectedRange>,
    },
}

/// The range we expect to read. If it's different from this, assert failure.
pub struct ExpectedRange {
    pub expected_start: u64,
    pub expected_end: u64,
}

/// A response for use with `httptest` which is aware of HTTP ranges.
pub struct RangeAwareResponse(u16, RangeAwareResponseType);

impl RangeAwareResponse {
    /// Create a new response suitable for use with `httptest` which can
    /// reply with ranges.
    pub fn new(status_code: u16, body: RangeAwareResponseType) -> Self {
        Self(status_code, body)
    }
}

impl httptest::responders::Responder for RangeAwareResponse {
    fn respond<'a>(
        &mut self,
        req: &'a httptest::http::Request<httptest::bytes::Bytes>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = httptest::http::Response<hyper::Body>> + Send + 'a>,
    > {
        let mut builder = http::Response::builder();
        builder = builder
            .status(StatusCode::from_u16(self.0).unwrap())
            .version(Version::HTTP_2)
            .header("Accept-Ranges", "bytes");
        let (body, content_length): (Option<hyper::body::Bytes>, _) = match &self.1 {
            RangeAwareResponseType::LengthOnly(len) => (None, *len),
            RangeAwareResponseType::Body {
                body,
                expected_range,
            } => {
                let (body, content_length) =
                    if let Some(range) = req.headers().get(http::header::RANGE) {
                        let range_regex = Regex::new(r"bytes=(\d+)-(\d+)").unwrap();
                        if let Some(captures) = range_regex.captures(range.to_str().unwrap()) {
                            let from = captures
                                .get(1)
                                .and_then(|s| s.as_str().parse::<usize>().ok())
                                .unwrap();
                            let to = captures
                                .get(2)
                                .and_then(|s| s.as_str().parse::<usize>().ok())
                                .unwrap();

                            if let Some(expected_range) = expected_range {
                                assert_eq!(
                                    expected_range.expected_start, from as u64,
                                    "Unexpected start location"
                                );
                                assert_eq!(
                                    expected_range.expected_end, to as u64,
                                    "Unexpected end location"
                                );
                            }
                            (body.slice(from..to), to - from)
                        } else {
                            assert!(expected_range.is_none());
                            (body.clone(), body.len())
                        }
                    } else {
                        assert!(expected_range.is_none());
                        (body.clone(), body.len())
                    };
                (Some(body), content_length)
            }
        };
        builder
            .header("Content-Length", format!("{content_length}"))
            .body(body.unwrap_or_default())
            .unwrap()
            .respond(req)
    }
}

/// For testing purposes, an `httptest` server configured to respond in particular
/// ways.
#[derive(Debug, Clone, Copy, strum::Display, Arbitrary, strum::EnumIter)]
#[strum(serialize_all = "snake_case")]
pub enum ServerType {
    /// This server doesn't supply a `Content-Length` header
    NoContentLength,
    /// This server supplies `Content-Length` but does not support range requests.
    ContentLengthButNoRanges,
    /// This server fully supports HTTP ranges.
    Ranges,
}

impl ServerType {
    /// Get an iterator for all the different server types.
    pub fn types() -> impl Iterator<Item = ServerType> {
        Self::iter()
    }
}

/// Set up an `httptest` server to respond according to the given [`ServerType`].
pub fn set_up_server(server: &Server, zip_data: Vec<u8>, server_type: ServerType) {
    match server_type {
        ServerType::NoContentLength => {
            server.expect(
                Expectation::matching(httptest::matchers::request::method_path("HEAD", "/foo"))
                    .times(..)
                    .respond_with(status_code(200)),
            );
            server.expect(
                Expectation::matching(httptest::matchers::request::method_path("GET", "/foo"))
                    .times(..)
                    .respond_with(status_code(200).body(zip_data)),
            );
        }
        ServerType::ContentLengthButNoRanges => {
            server.expect(
                Expectation::matching(httptest::matchers::request::method_path("HEAD", "/foo"))
                    .times(..)
                    .respond_with(
                        status_code(200)
                            .append_header("Content-Length", format!("{}", zip_data.len())),
                    ),
            );
            server.expect(
                Expectation::matching(httptest::matchers::request::method_path("GET", "/foo"))
                    .times(..)
                    .respond_with(status_code(200).body(zip_data)),
            );
        }
        ServerType::Ranges => {
            server.expect(
                Expectation::matching(httptest::matchers::request::method_path("HEAD", "/foo"))
                    .times(..)
                    .respond_with(RangeAwareResponse::new(
                        200,
                        RangeAwareResponseType::LengthOnly(zip_data.len()),
                    )),
            );
            server.expect(
                Expectation::matching(httptest::matchers::request::method_path("GET", "/foo"))
                    .times(..)
                    .respond_with(RangeAwareResponse::new(
                        206,
                        RangeAwareResponseType::Body {
                            body: hyper::body::Bytes::from(zip_data),
                            expected_range: None,
                        },
                    )),
            );
        }
    }
}

/// How big to make files in a generated sample zip file.
#[derive(Clone, Copy, strum::Display, Eq, PartialEq, Hash)]
pub enum FileSize {
    Small,
    Medium,
    Big,
}

/// Whether all the file sizes in a generated sample zip file should be the
/// same or different sizes.
#[derive(Eq, PartialEq, Hash, Clone)]
pub enum FileSizes {
    Fixed(FileSize),
    Variable,
}

/// Parameters to be used in creating a new sample zip file.
#[derive(Eq, Clone)]
pub struct ZipParams {
    file_sizes: FileSizes,
    num_files: usize,
    compression: zip::CompressionMethod,
}

impl PartialEq for ZipParams {
    fn eq(&self, other: &Self) -> bool {
        // Manually specified to match the `Hash` implementation.
        self.file_sizes == other.file_sizes
            && self.num_files == other.num_files
            && self.compression == other.compression
    }
}

impl std::hash::Hash for ZipParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Necessary because zip::CompressionMethod is not
        // `Hash`
        self.file_sizes.hash(state);
        self.num_files.hash(state);
        let compression_int = match self.compression {
            zip::CompressionMethod::Stored => 0,
            zip::CompressionMethod::Deflated => 1,
            zip::CompressionMethod::Bzip2 => 2,
            zip::CompressionMethod::Aes => 3,
            zip::CompressionMethod::Zstd => 4,
            _ => todo!(),
        };
        compression_int.hash(state);
    }
}

impl ZipParams {
    pub fn new(
        file_sizes: FileSizes,
        num_files: usize,
        compression: zip::CompressionMethod,
    ) -> Self {
        Self {
            file_sizes,
            num_files,
            compression,
        }
    }
}

impl Display for ZipParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ", self.num_files)?;
        match self.file_sizes {
            FileSizes::Fixed(sz) => write!(f, "{}", sz.to_string().to_lowercase())?,
            FileSizes::Variable => write!(f, "variable")?,
        };
        let comp = match self.compression {
            zip::CompressionMethod::Stored => "stored",
            zip::CompressionMethod::Deflated => "deflated",
            _ => todo!(),
        };
        write!(f, " {comp}")
    }
}

fn file_generator(file_size: FileSize) -> impl FnMut() -> String {
    move || {
        lipsum::lipsum(match file_size {
            FileSize::Small => 25,
            FileSize::Medium => 20000,
            FileSize::Big => 1000000,
        })
    }
}

/// Creat a zip file of a certain nature
fn create_zip(w: impl Write + Seek, zip_params: &ZipParams) {
    let mut zip = ZipWriter::new(w);

    let options = FileOptions::default()
        .compression_method(zip_params.compression)
        .unix_permissions(0o755);

    let mut file_generator: Box<dyn Iterator<Item = _>> = match zip_params.file_sizes {
        FileSizes::Fixed(size) => Box::new(std::iter::repeat(size).map(file_generator)),
        FileSizes::Variable => Box::new(
            std::iter::repeat(
                [FileSize::Small, FileSize::Medium, FileSize::Big]
                    .into_iter()
                    .map(file_generator),
            )
            .flatten(),
        ),
    };

    for i in 0..zip_params.num_files {
        zip.start_file(format!("{i}.txt"), options).unwrap();
        zip.write_all(file_generator.next().unwrap()().as_bytes())
            .unwrap();
    }

    zip.finish().unwrap();
}

type ZipMap = HashMap<ZipParams, Vec<u8>>;

/// Cache of our sample zips to avoid recreating in the setup for multiple benches - it's quite expensive.
static SAMPLE_ZIP_STORE: Lazy<Mutex<RefCell<ZipMap>>> =
    Lazy::new(|| Mutex::new(RefCell::new(HashMap::new())));

/// Get some sample zip data of a particular type. Lazily creates the zip
/// data requested then stores it for re-use.
pub fn get_sample_zip(zip_params: &ZipParams) -> Vec<u8> {
    let sample_zip_store = SAMPLE_ZIP_STORE.lock().unwrap();
    let mut sample_zip_store = sample_zip_store.borrow_mut();
    sample_zip_store
        .entry(zip_params.clone())
        .or_insert_with(|| {
            let mut data = Vec::new();
            create_zip(Cursor::new(&mut data), zip_params);
            data
        })
        .clone()
}
