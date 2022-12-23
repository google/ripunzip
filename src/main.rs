// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{fmt::Write, fs::File, path::PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use ripunzip::{UnzipEngine, UnzipOptions, UnzipProgressReporter};

/// Unzip all files within a zip file as quickly as possible.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,

    /// The output directory into which to place the files. By default, the
    /// current working directory is used.
    #[arg(short = 'd', long, value_name = "DIRECTORY")]
    output_directory: Option<PathBuf>,

    /// Whether to decompress on a single thread. By default,
    /// multiple threads are used, but this can lead to more network traffic.
    #[arg(long)]
    single_threaded: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// unzips a zip file
    File {
        /// Zip file to unzip
        #[arg(value_name = "FILE")]
        zipfile: PathBuf,
    },
    /// downloads and unzips a zip file
    Uri {
        /// URI of zip file to download and unzip
        #[arg(value_name = "URI")]
        uri: String,

        /// Limit how far in the zip file we read ahead to allow parallel unzips.
        /// By default, this is unlimited, which means total RAM use of this tool can be as much as the
        /// fully compressed file size (in pathological cases only). Adding this limit will solve that
        /// problem, but may make transfers much less efficient by requiring multiple HTTP streams.
        #[arg(long, value_name = "BYTES")]
        readahead_limit: Option<usize>,
    },
}

fn main() -> Result<()> {
    use std::io::Write;

    env_logger::builder()
        .format(|buf, record| {
            let ts = buf.timestamp_micros();
            writeln!(
                buf,
                "{}: {:?}: {}: {}",
                ts,
                std::thread::current().id(),
                buf.default_level_style(record.level())
                    .value(record.level()),
                record.args()
            )
        })
        .init();
    let args = Args::parse();
    let options = UnzipOptions {
        output_directory: args.output_directory,
        single_threaded: args.single_threaded,
    };
    match &args.command {
        Commands::File { zipfile } => {
            let zipfile = File::open(zipfile)?;
            UnzipEngine::for_file(zipfile, options, ProgressDisplayer::new())?.unzip()
        }
        Commands::Uri {
            uri,
            readahead_limit,
        } => UnzipEngine::for_uri(
            uri,
            options,
            *readahead_limit,
            ProgressDisplayer::new(),
            report_on_insufficient_readahead_size,
        )?
        .unzip(),
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
        self.0.set_message(format!("Extracting {}", display_name))
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
