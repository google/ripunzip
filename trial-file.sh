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

# This is a noddy benchmark to compare regular unzip to ripunzip.
# It may not be realistic.

ZIPFILE="$1"
MYDIR="$(pwd)"

echo Ensure file cache is warmed up to contain zipfile
cat $ZIPFILE > /dev/null

rm -Rf /tmp/testb
mkdir /tmp/testb
pushd /tmp/testb
echo ripunzip:
time sh -c "$MYDIR/target/release/ripunzip $ZIPFILE > /dev/null"
popd
rm -Rf /tmp/testb

rm -Rf /tmp/testa
mkdir /tmp/testa
pushd /tmp/testa
echo unzip:
time sh -c "unzip $ZIPFILE > /dev/null"
popd
rm -Rf /tmp/testa
