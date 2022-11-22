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

use std::{
    fs::{create_dir_all, File},
    io::SeekFrom,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::io::prelude::*;

/// Unzip all files within a zip file as quickly as possible.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Zip file to unzip
    #[arg(value_name = "FILE")]
    zipfile: PathBuf,
}

#[derive(Clone)]
struct CloneableFile {
    file: Arc<Mutex<File>>,
    pos: u64,
    // TODO determine and store this once instead of per cloneable file
    file_length: Option<u64>,
}

impl<'a> CloneableFile {
    fn new(file: File) -> Self {
        Self {
            file: Arc::new(Mutex::new(file)),
            pos: 0u64,
            file_length: None,
        }
    }
}

impl CloneableFile {
    fn ascertain_file_length(&mut self) -> u64 {
        match self.file_length {
            Some(file_length) => file_length,
            None => {
                let len = self.file.lock().unwrap().metadata().unwrap().len();
                println!("File length determined to be {}", len);
                self.file_length = Some(len);
                len
            }
        }
    }
}

impl Read for CloneableFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut underlying_file = self.file.lock().expect("Unable to get underlying file");
        // TODO share an object which knows current position to avoid unnecessary
        // seeks
        underlying_file.seek(SeekFrom::Start(self.pos))?;
        let read_result = underlying_file.read(buf);
        if let Ok(bytes_read) = read_result {
            // TODO, once stabilised, use checked_add_signed
            self.pos += bytes_read as u64;
        }
        read_result
    }
}

impl Seek for CloneableFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(pos) => pos,
            SeekFrom::End(offset_from_end) => {
                let file_len = self.ascertain_file_length();
                // TODO, once stabilised, use checked_add_signed
                file_len - (-offset_from_end as u64)
            }
            // TODO, once stabilised, use checked_add_signed
            SeekFrom::Current(offset_from_pos) => {
                if offset_from_pos == 0 {
                    self.pos
                } else if offset_from_pos > 0 {
                    self.pos + (offset_from_pos as u64)
                } else {
                    self.pos - ((-offset_from_pos) as u64)
                }
            }
        };
        self.pos = new_pos;
        Ok(new_pos)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let zipfile = File::open(args.zipfile)?;
    let zipfile = CloneableFile::new(zipfile);
    let zip = zip::ZipArchive::new(zipfile)?;
    let file_count = zip.len();
    println!("Zip has {} files", file_count);
    (0..file_count).into_par_iter().for_each(|i| {
        let mut myzip = zip.clone();
        let mut file = myzip.by_index(i).expect("Unable to get file from zip");
        let name = file.name();
        println!("Filename: {}", name);
        if name.ends_with("/") {
            println!("Skipping, directory");
        } else {
            let out_file = PathBuf::from(file.name());
            if let Some(parent) = out_file.parent() {
                create_dir_all(parent).unwrap_or_else(|err| {
                    panic!("Unable to create parent directories for {}: {}", name, err)
                });
            }
            let mut out_file = File::create(out_file)
                .unwrap();
            std::io::copy(&mut file, &mut out_file).unwrap();
        }
    });
    Ok(())
}
