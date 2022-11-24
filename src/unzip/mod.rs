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

use std::{
    borrow::Cow,
    fs::{create_dir_all, File},
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use rayon::prelude::*;
use zip::{read::ZipFile, ZipArchive};

use crate::unzip::cloneable_seekable_reader::CloneableSeekableReader;

/// Options for unzipping.
pub struct UnzipOptions {
    /// The destination directory.
    pub output_directory: Option<PathBuf>,
}

/// Download a zip file from a URI and unzip it. At the moment, this
/// does the download and unzipping in sequence, but in the future it's
/// hoped this library will evolve to do them in parallel.
/// At the moment, the result is an [`anyhow::Result`] which is not
/// the most appropriate for a library; we may switch to `thiserror`
/// eventually.
pub fn unzip_uri(uri: &str, options: &UnzipOptions) -> Result<()> {
    println!("Downloading URI {}", uri);
    let mut response = reqwest::blocking::get(uri)?;
    let mut tempfile = tempfile::tempfile()?;
    std::io::copy(&mut response, &mut tempfile)?;
    unzip_file(tempfile, options)
}

/// Unzip a file, optionally specifying an output directory.
/// This will attempt to unzip the file in a multi-threaded fashion.
/// At the moment, the result is an [`anyhow::Result`] which is not
/// the most appropriate for a library; we may switch to `thiserror`
/// eventually.
pub fn unzip_file(zipfile: File, options: &UnzipOptions) -> Result<()> {
    let output_directory = &options.output_directory;
    // The following line doesn't actually seem to make any significant
    // performance difference.
    // let zipfile = BufReader::new(zipfile);
    let zipfile = CloneableSeekableReader::new(zipfile);
    let zip = ZipArchive::new(zipfile)?;
    let file_count = zip.len();
    println!("Zip has {} files", file_count);
    let errors: Vec<_> = (0..file_count)
        .into_par_iter()
        .map(|i| extract_file(zip.clone(), i, output_directory))
        .filter_map(Result::err)
        .collect();
    // Output any errors we found on any file
    for error in &errors {
        eprintln!("Error: {}", error)
    }
    // Return the first error code, if any.
    errors.into_iter().next().map(Result::Err).unwrap_or(Ok(()))
}

/// Extracts a file from a zip file, attaching diagnostics to any errors where
/// possible.
fn extract_file(
    mut myzip: zip::ZipArchive<CloneableSeekableReader<File>>,
    i: usize,
    output_directory: &Option<PathBuf>,
) -> Result<()> {
    let file = myzip.by_index(i)?;
    let name = file
        .enclosed_name()
        .map(Path::to_string_lossy)
        .unwrap_or_else(|| Cow::Borrowed("<unprintable>"))
        .to_string();
    extract_file_inner(file, output_directory)
        .with_context(|| format!("Failed to extract {}", name))
}

/// Extracts a file from a zip file.
fn extract_file_inner(mut file: ZipFile, output_directory: &Option<PathBuf>) -> Result<()> {
    if file.is_dir() {
        return Ok(());
    }
    let name = file
        .enclosed_name()
        .ok_or_else(|| std::io::Error::new(ErrorKind::Unsupported, "path not safe to extract"))?;
    let name = name.to_path_buf();
    println!("Extracting: {}", name.display());
    let out_file = match output_directory {
        Some(output_directory) => output_directory.join(file.name()),
        None => PathBuf::from(file.name()),
    };
    if let Some(parent) = out_file.parent() {
        create_dir_all(parent)?;
    }
    let mut out_file = File::create(out_file)?;
    std::io::copy(&mut file, &mut out_file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        env::{current_dir, set_current_dir},
        fs::{read_to_string, File},
        io::Write,
        path::Path,
    };

    use tempfile::tempdir;
    use zip::{write::FileOptions, ZipWriter};

    use crate::UnzipOptions;

    use super::unzip_file;

    fn create_zip_file(path: &Path) {
        let file = File::create(path).unwrap();
        let mut zip = ZipWriter::new(file);

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
        };
        unzip_file(zf, &options).unwrap();
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
        };
        unzip_file(zf, &options).unwrap();
        check_files_exist(&outdir);
    }
}
