#!/bin/bash -l

export arg=$1
export testdir=$(dirname $0)
export hercules="$arg/hercules"

# argv test
[ "$($hercules run "$testdir/argv.hs" aa bb cc)" == "aa,bb,cc" ] || exit 1

# build test
$hercules build -release -o "$arg/test_binary" "$testdir/build.hs"
[ "$($arg/test_binary)" == "hello" ] || exit 2

# library test
$hercules build -relocation-model=pic -o "$arg/libhercules_export_test.so" "$testdir/export.hs"
gcc "$testdir/test.c" -L"$arg" -Wl,-rpath,"$arg" -lhercules_export_test -o "$arg/test_binary"
[ "$($arg/test_binary)" == "abcabcabc" ] || exit 3

# exit code test
$hercules run "$testdir/exit.hs" || if [[ $? -ne 42 ]]; then exit 4; fi
