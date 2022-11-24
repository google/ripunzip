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

use std::{fs::File, path::PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand};
use ripunzip::{unzip_file, unzip_uri, UnzipOptions};

/// Unzip all files within a zip file as quickly as possible.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, value_name = "DIRECTORY")]
    output_directory: Option<PathBuf>,
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
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    let options = UnzipOptions {
        output_directory: args.output_directory,
    };
    match &args.command {
        Commands::File { zipfile } => {
            let zipfile = File::open(zipfile)?;
            unzip_file(zipfile, &options)
        }
        Commands::Uri { uri } => unzip_uri(uri, &options),
    }
}
