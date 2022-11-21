use std::{
    fs::File,
    io::SeekFrom,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Result};
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
    pos: Option<u64>,
    file_length: Option<u64>,
}

impl<'a> CloneableFile {
    fn new(file: File) -> Self {
        Self {
            file: Arc::new(Mutex::new(file)),
            pos: None,
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
                self.file_length = Some(len);
                len
            }
        }
    }
}

impl Read for CloneableFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut underlying_file = self.file.lock().expect("Unable to get underlying file");
        if let Some(seek_from) = self.pos {
            underlying_file.seek(SeekFrom::Start(seek_from))?;
        }
        underlying_file.read(buf)
    }
}

impl Seek for CloneableFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let prior_pos = self.pos.unwrap_or_default();
        let new_pos = match pos {
            SeekFrom::Start(pos) => pos,
            SeekFrom::End(offset_from_end) => {
                let file_len = self.ascertain_file_length();
                // TODO, once stabilised, use checked_add_signed
                file_len - (offset_from_end as u64)
            }
            // TODO, once stabilised, use checked_add_signed
            SeekFrom::Current(offset_from_pos) => {
                if offset_from_pos > 0 {
                    prior_pos + (offset_from_pos as u64)
                } else {
                    prior_pos - ((-offset_from_pos) as u64)
                }
            }
        };
        self.pos = Some(new_pos);
        Ok(new_pos)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let zipfile = File::open(args.zipfile)?;
    let zipfile = CloneableFile::new(zipfile);
    let zip = zip::ZipArchive::new(zipfile)?;
    let file_count = zip.len();
    (0..file_count).into_par_iter().for_each(|i| {
        let mut myzip = zip.clone();
        let file = myzip.by_index(i).expect("Unable to get file from zip");
        println!("Filename: {}", file.name());
        //std::io::copy(&mut file, &mut std::io::stdout());
    });
    println!("Hello, world!");
    Ok(())
}
