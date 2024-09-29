#!/bin/sh

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This is a noddy benchmark to compare regular unzip to ripunzip.
# It may not be realistic. For something with more statistical
# rigor, see the benches directory.

URI="https://chromium-browser-asan.storage.googleapis.com/linux-release/asan-linux-release-970006.zip"

MYDIR="$(pwd)"

rm -Rf /tmp/testb
mkdir /tmp/testb
pushd /tmp/testb
echo ripunzip:
time sh -c "$MYDIR/target/release/ripunzip unzip-uri \"$URI\" "
popd
rm -Rf /tmp/testb

