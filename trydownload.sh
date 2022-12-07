#!/bin/bash

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Weirdly this supports ranges even though https://commondatastorage.googleapis.com/chromium-browser-asan/index.html?prefix=linux-release/asan-linux-release-97 doesn't.
cargo run --release -- -o out uri 'https://chromium-browser-asan.storage.googleapis.com/linux-release/asan-linux-release-970006.zip' 
