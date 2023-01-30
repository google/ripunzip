// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use criterion::{criterion_group, criterion_main, Criterion};
use ripunzip_test_utils::*;
use std::string::ToString;
use tempfile::TempDir;

fn ripunzip_path() -> PathBuf {
    std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("ripunzip")
}

fn create_output_dir() -> tempfile::TempDir {
    let output_dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(&output_dir).unwrap();
    output_dir
}

fn create_output_dir_and_zip_file(
    params: &ZipParams,
) -> (tempfile::TempDir, tempfile::NamedTempFile) {
    let output_dir = create_output_dir();
    let mut zip_file = tempfile::NamedTempFile::new().unwrap();
    let zip_data = get_sample_zip(params);
    zip_file.write_all(&zip_data).unwrap();
    (output_dir, zip_file)
}

fn file_comparison(c: &mut Criterion, params: &ZipParams) {
    let desc = format!("file {params}",);
    let ripunzip_path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("ripunzip");

    let mut group = c.benchmark_group(&desc);
    group.bench_function(&format!("{} ripunzip", &desc), |b| {
        b.iter_batched(
            || create_output_dir_and_zip_file(params),
            |(output_dir, zip_file)| {
                let zip_file_path = zip_file.path();
                Command::new(&ripunzip_path)
                    .arg("-d")
                    .arg(output_dir.path())
                    .arg("file")
                    .arg(zip_file_path)
                    .output()
                    .expect("failed to ripunzip");
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function(&format!("{} unzip", &desc), |b| {
        b.iter_batched(
            || create_output_dir_and_zip_file(params),
            |(output_dir, zip_file)| {
                let zip_file_path = zip_file.path();
                Command::new("unzip")
                    .arg("-d")
                    .arg(output_dir.path())
                    .arg(zip_file_path)
                    .output()
                    .expect("failed to unzip");
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn uri_comparison(c: &mut Criterion, params: &ZipParams, server_type: ServerType) {
    let desc = format!("uri {server_type} {params}",);

    let create_output_dir_and_server = move || {
        let output_dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();
        let zip_data = get_sample_zip(params);

        let server = httptest::Server::run();
        set_up_server(&server, zip_data, server_type);

        (output_dir, server)
    };
    let ripunzip_path = ripunzip_path();

    let mut group = c.benchmark_group(&desc);
    group.bench_function(&format!("{} ripunzip", &desc), |b| {
        b.iter_batched(
            create_output_dir_and_server,
            |(output_dir, server)| {
                let uri = &server.url("/foo").to_string();
                fetch_uri_with_ripunzip(uri, output_dir, &ripunzip_path);
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function(&format!("{} unzip", &desc), |b| {
        b.iter_batched(
            create_output_dir_and_server,
            |(output_dir, server)| {
                let uri = &server.url("/foo").to_string();
                fetch_uri_with_curl_and_unzip(uri, output_dir)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

#[cfg(feature = "real_world_benchmark")]
fn real_world_uri_comparison(c: &mut Criterion) {
    const URI: &str = "https://chromium-browser-asan.storage.googleapis.com/linux-release/asan-linux-release-970006.zip";
    let desc = "realuri";

    let ripunzip_path = ripunzip_path();

    let mut group = c.benchmark_group(desc);
    group.bench_function(&format!("{} ripunzip", &desc), |b| {
        b.iter_batched(
            create_output_dir.clone(),
            |output_dir| {
                fetch_uri_with_ripunzip(URI, output_dir, &ripunzip_path);
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function(&format!("{} unzip", &desc), |b| {
        b.iter_batched(
            create_output_dir.clone(),
            |output_dir| fetch_uri_with_curl_and_unzip(URI, output_dir),
            criterion::BatchSize::SmallInput,
        )
    });
}

#[cfg(not(feature = "real_world_benchmark"))]
fn real_world_uri_comparison(_c: &mut Criterion) {}

fn fetch_uri_with_ripunzip(uri: &str, output_dir: TempDir, ripunzip_path: &Path) {
    Command::new(ripunzip_path)
        .arg("-d")
        .arg(output_dir.path())
        .arg("uri")
        .arg(uri)
        .output()
        .expect("failed to ripunzip");
}

fn fetch_uri_with_curl_and_unzip(uri: &str, output_dir: TempDir) {
    let downloaded_zip = output_dir.path().join("file.zip");
    Command::new("curl")
        .arg("-o")
        .arg(&downloaded_zip)
        .arg(uri)
        .output()
        .expect("curl failed");
    let inner_output_dir = output_dir.path().join("real_output");
    std::fs::create_dir_all(&inner_output_dir).unwrap();
    Command::new("unzip")
        .arg("-d")
        .arg(inner_output_dir)
        .arg(&downloaded_zip)
        .output()
        .expect("failed to unzip");
}

fn add_benchmarks(c: &mut Criterion) {
    let zips_to_test = [
        ZipParams::new(
            FileSizes::Fixed(FileSize::Small),
            100,
            zip::CompressionMethod::Stored,
        ),
        ZipParams::new(FileSizes::Variable, 100, zip::CompressionMethod::Stored),
        ZipParams::new(
            FileSizes::Fixed(FileSize::Small),
            100,
            zip::CompressionMethod::Deflated,
        ),
        ZipParams::new(FileSizes::Variable, 100, zip::CompressionMethod::Deflated),
        ZipParams::new(FileSizes::Variable, 1000, zip::CompressionMethod::Deflated),
    ];
    real_world_uri_comparison(c);
    for z in zips_to_test.iter() {
        file_comparison(c, z);
        for server_type in ServerType::types() {
            uri_comparison(c, z, server_type);
        }
    }
}

criterion_group!(name = benches; config = Criterion::default().sample_size(10); targets = add_benchmarks);
criterion_main!(benches);
