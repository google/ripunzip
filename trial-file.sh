#!/bin/sh

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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
time sh -c "$MYDIR/target/release/ripunzip file $ZIPFILE > /dev/null"
popd
rm -Rf /tmp/testb

rm -Rf /tmp/testa
mkdir /tmp/testa
pushd /tmp/testa
echo unzip:
time sh -c "unzip $ZIPFILE > /dev/null"
popd
rm -Rf /tmp/testa
