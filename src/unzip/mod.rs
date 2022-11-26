// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod cloneable_seekable_reader;
mod http_range_reader;
mod seekable_http_reader;
#[cfg(test)]
mod test_utils;

use std::{
    borrow::Cow,
    fs::{create_dir_all, File, Permissions},
    io::{ErrorKind, Read, Seek},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result};
use rayon::prelude::*;
use zip::{read::ZipFile, result::ZipResult, ZipArchive};

use crate::unzip::cloneable_seekable_reader::CloneableSeekableReader;

use self::{
    cloneable_seekable_reader::HasLength,
    seekable_http_reader::{SeekableHttpReader, SeekableHttpReaderEngine},
};

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
    ) -> Vec<anyhow::Error> {
        if single_threaded {
            (0..self.len())
                .into_iter()
                .map(|i| extract_file(&mut self.0, i, output_directory, progress_reporter))
                .filter_map(Result::err)
                .collect()
        } else {
            (0..self.len())
                .into_par_iter()
                .map(|i| extract_file(&mut self.0.clone(), i, output_directory, progress_reporter))
                .filter_map(Result::err)
                .collect()
        }
    }
}

/// Engine which knows how to unzip a URI.
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
    ) -> Vec<anyhow::Error> {
        self.0.create_reader_at_zero();
        let result = if single_threaded {
            (0..self.len())
                .into_iter()
                .map(|i| extract_file(&mut self.1, i, output_directory, progress_reporter))
                .filter_map(Result::err)
                .collect()
        } else {
            (0..self.len())
                .into_par_iter()
                .map(|i| extract_file(&mut self.1.clone(), i, output_directory, progress_reporter))
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
        })
    }

    /// Create an unzip engine which knows how to unzip a URI.
    /// Parameters:
    /// - the URI
    /// - unzip options
    /// - how big a readahead buffer to create in memory.
    /// - a progress reporter (set of callbacks)
    /// - an additional callback to warn if performance was impaired by
    ///   rewinding the HTTP stream. (This implies the readahead buffer was
    ///   too small.)
    pub fn for_uri<F: Fn() + 'static>(
        uri: &str,
        options: UnzipOptions,
        readahead_limit: Option<usize>,
        progress_reporter: P,
        callback_on_rewind: F,
    ) -> Result<Self> {
        let seekable_http_reader = SeekableHttpReaderEngine::new(uri.to_string(), readahead_limit);
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
        let errors = self
            .zipfile
            .unzip(single_threaded, output_directory, &self.progress_reporter);
        // Output any errors we found on any file
        for error in &errors {
            eprintln!("Error: {}", error)
        }
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
) -> Result<()> {
    let file = myzip.by_index(i)?;
    let name = file
        .enclosed_name()
        .map(Path::to_string_lossy)
        .unwrap_or_else(|| Cow::Borrowed("<unprintable>"))
        .to_string();
    extract_file_inner(file, output_directory, progress_reporter)
        .with_context(|| format!("Failed to extract {}", name))
}

/// Extracts a file from a zip file.
fn extract_file_inner(
    mut file: ZipFile,
    output_directory: &Option<PathBuf>,
    progress_reporter: &dyn UnzipProgressReporter,
) -> Result<()> {
    if file.is_dir() {
        return Ok(());
    }
    let name = file
        .enclosed_name()
        .ok_or_else(|| std::io::Error::new(ErrorKind::Unsupported, "path not safe to extract"))?;
    let name = name.to_path_buf();
    let display_name = name.display().to_string();
    progress_reporter.extraction_starting(&display_name);
    let out_path = match output_directory {
        Some(output_directory) => output_directory.join(file.name()),
        None => PathBuf::from(file.name()),
    };
    if let Some(parent) = out_path.parent() {
        create_dir_all(parent)?;
    }
    let mut out_file = File::create(&out_path)?;
    std::io::copy(&mut file, &mut out_file)?;
    progress_reporter.bytes_extracted(file.compressed_size());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Some(mode) = file.unix_mode() {
            std::fs::set_permissions(&out_path, Permissions::from_mode(mode))?;
        }
    }
    progress_reporter.extraction_finished(&display_name);
    Ok(())
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

    use crate::{NullProgressReporter, UnzipEngine, UnzipOptions};

    use super::test_utils::RangeAwareResponse;

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

    use httptest::{matchers::request::method_path, Expectation, Server};

    #[test]
    fn test_extract_from_server() {
        let td = tempdir().unwrap();
        let mut zip_data = Cursor::new(Vec::new());
        create_zip(&mut zip_data);
        let body = zip_data.into_inner();
        println!("Whole zip:");
        hexdump::hexdump(&body);

        let server = Server::run();
        server.expect(
            Expectation::matching(method_path("HEAD", "/foo"))
                .times(..)
                .respond_with(RangeAwareResponse::new(200, body.clone())),
        );
        server.expect(
            Expectation::matching(method_path("GET", "/foo"))
                .times(..)
                .respond_with(RangeAwareResponse::new(206, body)),
        );

        let outdir = td.path().join("outdir");
        let options = UnzipOptions {
            output_directory: Some(outdir.clone()),
            single_threaded: false,
        };
        UnzipEngine::for_uri(
            &server.url("/foo").to_string(),
            options,
            None,
            NullProgressReporter,
            || {},
        )
        .unwrap()
        .unzip()
        .unwrap();
        check_files_exist(&outdir);
    }
}
