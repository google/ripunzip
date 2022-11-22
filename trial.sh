#!/bin/sh

# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ZIPFILE=~/Downloads/mac-release_asan-mac-release-1008971.zip
MYDIR=$(pwd)

mkdir /tmp/testa
pushd /tmp/testa
time sh -c "unzip $ZIPFILE > /dev/null"
popd

rm -Rf /tmp/testa

mkdir /tmp/testb
pushd /tmp/testb
time sh -c "$MYDIR/target/release/runzip $ZIPFILE > /dev/null"
popd

rm -Rf /tmp/testb
