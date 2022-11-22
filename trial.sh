#!/bin/sh

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
