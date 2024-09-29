// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(unsafe_code)]

use std::{fmt::Write, fs::File, path::PathBuf, sync::RwLock};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use ripunzip::{
    FilenameFilter, NullProgressReporter, UnzipEngine, UnzipOptions, UnzipProgressReporter,
};
use wildmatch::WildMatch;

const LONG_ABOUT: &str =
    "ripunzip is a tool to unzip zip files in parallel, possibly from a remote server.
It works best with HTTP(S) servers that support Range requests.";

/// Unzip all files within a zip file as quickly as possible.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = LONG_ABOUT)]
struct RipunzipArgs {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Lists a zip file
    ListFile {
        #[command(flatten)]
        file_args: FileArgs,
    },

    /// Unzip a zip file
    UnzipFile {
        #[command(flatten)]
        file_args: FileArgs,

        #[command(flatten)]
        unzip_args: UnzipArgs,
    },

    /// Lists a zip file from a URI
    ListUri {
        #[command(flatten)]
        uri_args: UriArgs,
    },

    /// Unzips a zip file from a URI
    UnzipUri {
        #[command(flatten)]
        uri_args: UriArgs,

        #[command(flatten)]
        unzip_args: UnzipArgs,
    },
}

#[derive(Args, Debug)]
struct UnzipArgs {
    /// The output directory into which to place the files. By default, the
    /// current working directory is used.
    #[arg(short = 'd', long, value_name = "DIRECTORY")]
    output_directory: Option<PathBuf>,

    /// Password to decrypt encrypted zipfile entries (if any).
    /// Both ZipCrypto and AES encrypted zipfiles are supported.
    #[arg(short = 'P', long, value_name = "PASSWORD")]
    password: Option<String>,

    /// Whether to decompress on a single thread. By default,
    /// multiple threads are used, but this can lead to more network traffic.
    #[arg(long)]
    single_threaded: bool,

    /// Optionally, a list of files to unzip from the zip file. Omit
    /// to unzip all of them. This can include wildcards.
    #[arg(value_name = "FILES")]
    filenames_to_unzip: Vec<String>,
}

#[derive(Args, Debug)]
struct FileArgs {
    /// Zip file to unzip
    #[arg(value_name = "FILE")]
    zipfile: PathBuf,
}

#[derive(Args, Debug)]
struct UriArgs {
    /// URI of zip file to download and unzip
    #[arg(value_name = "URI")]
    uri: String,

    /// Limit how far in the zip file we read ahead to allow parallel unzips.
    /// By default, this is unlimited, which means total RAM use of this tool can be as much as the
    /// fully compressed file size (in pathological cases only). Adding this limit will solve that
    /// problem, but may make transfers much less efficient by requiring multiple HTTP streams.
    #[arg(long, value_name = "BYTES")]
    readahead_limit: Option<usize>,
}

fn main() -> Result<()> {
    let args = RipunzipArgs::parse();
    env_logger::Builder::new()
        .filter_level(args.verbose.log_level_filter())
        .init();
    match args.command {
        Commands::ListFile { file_args } => list(construct_file_engine(file_args)?),
        Commands::ListUri { uri_args } => list(construct_uri_engine(uri_args)?),
        Commands::UnzipFile {
            file_args,
            unzip_args,
        } => unzip(
            construct_file_engine(file_args)?,
            unzip_args,
            args.verbose.is_silent(),
        ),
        Commands::UnzipUri {
            uri_args,
            unzip_args,
        } => unzip(
            construct_uri_engine(uri_args)?,
            unzip_args,
            args.verbose.is_silent(),
        ),
    }
}

fn unzip(engine: UnzipEngine, unzip_args: UnzipArgs, is_silent: bool) -> Result<()> {
    let filename_filter: Option<Box<dyn FilenameFilter + Sync>> =
        if unzip_args.filenames_to_unzip.is_empty() {
            None
        } else {
            Some(Box::new(FileListFilter(RwLock::new(
                unzip_args
                    .filenames_to_unzip
                    .iter()
                    .map(|s| WildMatch::new(s))
                    .collect(),
            ))))
        };
    let progress_reporter: Box<dyn UnzipProgressReporter + Sync> = if is_silent {
        Box::new(NullProgressReporter)
    } else {
        Box::new(ProgressDisplayer::new())
    };
    let options = UnzipOptions {
        output_directory: unzip_args.output_directory,
        password: unzip_args.password,
        single_threaded: unzip_args.single_threaded,
        filename_filter,
        progress_reporter,
    };
    engine.unzip(options)
}

fn construct_file_engine(file_args: FileArgs) -> Result<UnzipEngine> {
    let zipfile = File::open(file_args.zipfile)?;
    UnzipEngine::for_file(zipfile)
}

fn construct_uri_engine(uri_args: UriArgs) -> Result<UnzipEngine> {
    UnzipEngine::for_uri(
        &uri_args.uri,
        uri_args.readahead_limit,
        report_on_insufficient_readahead_size,
    )
}

fn list(engine: UnzipEngine) -> Result<()> {
    let files = engine.list()?;
    for f in files {
        println!("{}", f);
    }
    Ok(())
}

struct FileListFilter(RwLock<Vec<WildMatch>>);

impl FilenameFilter for FileListFilter {
    fn should_unzip(&self, filename: &str) -> bool {
        let lock = self.0.read().unwrap();
        lock.iter().any(|m| m.matches(filename))
    }
}

fn report_on_insufficient_readahead_size() {
    eprintln!("Warning: this operation required several HTTP(S) streams.\nThis can slow down decompression.\nYou may wish to iuse --readahead-limit to increase the amount of data which can be held in memory.");
}

struct ProgressDisplayer(ProgressBar);

impl ProgressDisplayer {
    fn new() -> Self {
        Self(ProgressBar::new(0))
    }
}

impl UnzipProgressReporter for ProgressDisplayer {
    fn extraction_starting(&self, display_name: &str) {
        self.0.set_message(format!("Extracting {display_name}"))
    }

    fn total_bytes_expected(&self, expected: u64) {
        self.0.set_length(expected);
        self.0.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})\n{msg}")
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("#-"));
    }

    fn bytes_extracted(&self, count: u64) {
        self.0.inc(count)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::RwLock;

    use ripunzip::FilenameFilter;
    use wildmatch::WildMatch;

    use crate::FileListFilter;

    #[test]
    fn test_filelist_filter() {
        let patterns = &["dog", "cat", "mo?se", "ca*s"];
        let filter = FileListFilter(RwLock::new(
            patterns.iter().map(|p| WildMatch::new(p)).collect(),
        ));
        assert!(filter.should_unzip("dog"));
        assert!(!filter.should_unzip("dogs"));
        assert!(filter.should_unzip("cat"));
        assert!(filter.should_unzip("capybaras"));
        assert!(filter.should_unzip("cats"));
        assert!(filter.should_unzip("mouse"));
        assert!(filter.should_unzip("moose"));
        assert!(!filter.should_unzip("mouuuuuse"));
    }
}
