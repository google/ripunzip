# ripunzip

[![GitHub](https://img.shields.io/crates/l/ripunzip)](https://github.com/google/ripunzip)
[![crates.io](https://img.shields.io/crates/d/ripunzip)](https://crates.io/crates/ripunzip)
[![docs.rs](https://docs.rs/ripunzip/badge.svg)](https://docs.rs/ripunzip)

A tool to unzip files in parallel.

This is a Rust library (and command-line tool) which utilises the power of Rust's `rayon`
library to unzip a zip file in parallel. If you're fetching the zip file from a URI, it
may also be able to start unzipping in parallel with the download.

#### Installation and use

To fetch the command-line tool: `cargo install ripunzip` then `ripunzip -h`. Alternatively,
a `.deb` file is available under the "releases" section on github.

To add the library to your project: `cargo add ripunzip` and check out the documentation
linked above.

#### Development

Pull requests are welcome - see [the contributing doc](docs/contributing.md). The focus
of this project remains efficiently unzipping entire zip files, and any speed increases
are greatly appreciated! `cargo criterion` is used for performance testing, though the
benchmark suite doesn't do a great job of simulating real conditions. In particular please
be aware that this tool is often used on devices with spinny hard disks and very limited
disk write bandwidth, so in different circumstances that may be the limiting circumstance,
or network bandwidth, or CPU time. Please consider the impact of your changes on all these
permutations.

Release procedure:
1. Revise the version number
2. `cargo publish`
3. Retrieve the latest `.deb` file from the latest CI job
4. Declare a new release and tag on github
5. As you make that release, include the `.deb` file as an artifact.

There's also `cargo fuzz` support for comparitive fuzzing against non-parallel unzipping
to try to spot any unforeseen circumstances where we do anything differently. If you
change the core unzipping logic please use this.

#### License and usage notes

This is not an officially supported Google product.

<sup>
License

This software is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See LICENSE for details.
</sup>

