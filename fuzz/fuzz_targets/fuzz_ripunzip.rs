// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is a differential fuzzer for ripunzip's parallel unzip implementation
//! compared to the standard zip-rs behavior. Its purpose is not to test
//! the fundamentals of unzipping (as performed by zip-rs) but to check that
//! zip-rs and ripunzip always result in the same outcomes.

#![no_main]
use httptest::{responders::*, Expectation, Server};
use libfuzzer_sys::arbitrary;
use libfuzzer_sys::fuzz_target;
use ripunzip::RangeAwareResponse;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Cursor;
use std::io::Write;
use std::path::Path;
use itertools::Itertools;

/// Segments of the filename of each zip member. We don't allow
/// totally arbitrary filenames, because that tends to throw up permissions
/// errors, and our goal is not to fuzz the fundamental unzipping.
#[derive(arbitrary::Arbitrary, Debug, Clone, strum::Display, Hash, Eq, PartialEq)]
enum FilenameSegment {
    Fish,
    A,
    #[strum(serialize = "b")]
    B,
    #[strum(serialize = "31")]
    ThirtyOne,
    #[strum(serialize = "_c")]
    C,
    D,
    #[strum(serialize = "e.txt")]
    ETxt,
}

#[derive(arbitrary::Arbitrary, Eq, PartialEq, Hash, Debug, Clone)]
struct ZipMemberFilename(Vec<FilenameSegment>);

impl Into<String> for &ZipMemberFilename {
    fn into(self) -> String {
        self.0.iter().join("/")
    }
}

#[derive(Eq, PartialEq, Hash, Debug, Clone, Copy, arbitrary::Arbitrary)]
enum CompressionMethod {
    Stored,
    Deflated,
    Bzip2,
    Zstd,
}

#[derive(arbitrary::Arbitrary, Debug, Clone)]
enum ServerType {
    NoContentLength,
    ContentLengthButNoRanges,
    Ranges,
}

#[derive(arbitrary::Arbitrary, Debug, Clone)]
enum Method {
    File,
    Uri {
        readahead_limit: Option<usize>,
        server_type: ServerType,
    },
}

#[derive(arbitrary::Arbitrary, Debug, Clone)]
struct Inputs {
    // HashMap to ensure unique filenames in zip
    zip_members: HashMap<ZipMemberFilename, Vec<u8>>,
    compression_method: CompressionMethod,
    single_threaded: bool,
    method: Method,
}

fuzz_target!(|input: Inputs| {
    let progress_reporter = ripunzip::NullProgressReporter;
    let tempdir = tempfile::tempdir().unwrap();
    let output_directory = tempdir.path().join("out_ripunzip");
    let output_directory_unzip = tempdir.path().join("out_unzip");
    std::fs::create_dir_all(&output_directory).unwrap();
    let options = ripunzip::UnzipOptions {
        single_threaded: input.single_threaded,
        output_directory: Some(output_directory.clone()),
    };
    let zipfile = tempdir.path().join("file.zip");
    let mut zip_data = Vec::new();
    create_zip(&mut zip_data, &input.zip_members, input.compression_method);
    let mut file = std::fs::File::create(&zipfile).unwrap();
    file.write_all(&zip_data).unwrap();
    drop(file);
    let ripunzip_result: Result<(), anyhow::Error> = (|| match input.method {
        Method::File => {
            let file = std::fs::File::open(&zipfile).unwrap();
            let ripunzip = ripunzip::UnzipEngine::for_file(file, options, progress_reporter)?;
            ripunzip.unzip()
        }
        Method::Uri {
            readahead_limit,
            server_type,
        } => {
            let server = Server::run();
            match server_type {
                ServerType::NoContentLength => {
                    server.expect(
                        Expectation::matching(httptest::matchers::request::method_path(
                            "HEAD", "/foo",
                        ))
                        .times(..)
                        .respond_with(status_code(200)),
                    );
                    server.expect(
                        Expectation::matching(httptest::matchers::request::method_path(
                            "GET", "/foo",
                        ))
                        .times(..)
                        .respond_with(status_code(200).body(zip_data)),
                    );
                }
                ServerType::ContentLengthButNoRanges => {
                    server.expect(
                        Expectation::matching(httptest::matchers::request::method_path(
                            "HEAD", "/foo",
                        ))
                        .times(..)
                        .respond_with(
                            status_code(200)
                                .append_header("Content-Length", format!("{}", zip_data.len())),
                        ),
                    );
                    server.expect(
                        Expectation::matching(httptest::matchers::request::method_path(
                            "GET", "/foo",
                        ))
                        .times(..)
                        .respond_with(status_code(200).body(zip_data)),
                    );
                }
                ServerType::Ranges => {
                    server.expect(
                        Expectation::matching(httptest::matchers::request::method_path(
                            "HEAD", "/foo",
                        ))
                        .times(..)
                        .respond_with(RangeAwareResponse::new(200, zip_data.clone())),
                    );
                    server.expect(
                        Expectation::matching(httptest::matchers::request::method_path(
                            "GET", "/foo",
                        ))
                        .times(..)
                        .respond_with(RangeAwareResponse::new(206, zip_data)),
                    );
                }
            }
            let uri = &server.url("/foo").to_string();
            let ripunzip = ripunzip::UnzipEngine::for_uri(
                uri,
                options,
                readahead_limit,
                progress_reporter,
                || {},
            )?;
            ripunzip.unzip()
        }
    })();
    let unziprs_result = unzip_with_zip_rs(&zipfile, &output_directory_unzip);
    match unziprs_result {
        Err(err) => {
            if ripunzip_result.is_ok() {
                panic!("ripunzip succeeded; plain unzip gave {:?}", err)
            }
        }
        Ok(_) => {
            ripunzip_result.unwrap();
            let ripunzip_paths = recursive_lsdir(&output_directory);
            let unzip_paths = recursive_lsdir(&output_directory_unzip);
            // We do not currently compare the actual zip file contents.
            // It seems unlikely that this would be a failure mode where
            // ripunzip would differ from zip-rs.
            assert_eq!(ripunzip_paths, unzip_paths);
        }
    }
});

fn recursive_lsdir(dir: &Path) -> HashSet<std::path::PathBuf> {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.path().strip_prefix(dir).unwrap().to_path_buf())
        .collect()
}

fn create_zip(output: &mut Vec<u8>, zip_members: &HashMap<ZipMemberFilename, Vec<u8>>, compression_method: CompressionMethod) {
    let compression_method = match compression_method {
        CompressionMethod::Stored => zip::CompressionMethod::Stored,
        CompressionMethod::Deflated => zip::CompressionMethod::Deflated,
        CompressionMethod::Bzip2 => zip::CompressionMethod::Bzip2,
        CompressionMethod::Zstd => zip::CompressionMethod::Zstd,
    };
    let mut zip = zip::ZipWriter::new(Cursor::new(output));
    let options =
        zip::write::FileOptions::default().compression_method(compression_method);
    for (name, data) in zip_members.into_iter() {
        zip.start_file(name, options).unwrap();
        zip.write(&data).unwrap();
    }
    zip.finish().unwrap();
}

#[derive(Debug)]
enum ZipRsError {
    OpenFailed(std::io::Error),
    CreateExtractDirFailed(std::io::Error),
    ArchiveFailed(zip::result::ZipError),
    ExtractFailed(zip::result::ZipError),
}

/// Unzip the content with standard zip-rs.
fn unzip_with_zip_rs(zipfile_path: &Path, dest_path: &Path) -> Result<(), ZipRsError> {
    let mut zip =
        zip::ZipArchive::new(File::open(zipfile_path).map_err(|e| ZipRsError::OpenFailed(e))?)
            .map_err(|e| ZipRsError::ArchiveFailed(e))?;
    std::fs::create_dir_all(dest_path).map_err(|e| ZipRsError::CreateExtractDirFailed(e))?;
    zip.extract(dest_path)
        .map_err(|e| ZipRsError::ExtractFailed(e))
}
