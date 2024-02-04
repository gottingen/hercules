#!/usr/bin/env bash
set -e
set -o pipefail

export BENCH_DIR=$(dirname $0)
export PYTHON="${EXE_PYTHON:-python3}"
export PYPY="${EXE_PYPY:-pypy3}"
export CPP="${EXE_CPP:-g++}"
export HERCULES="${EXE_HERCULES:-build/hercules}"

echo "benchmark,python,pypy,cpp,hercules"

# MANDELBROT
echo -n "mandelbrot"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/mandelbrot/mandelbrot.py | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/mandelbrot/mandelbrot.py | tail -n 1)
echo -n ","
# nothing for cpp
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/mandelbrot/mandelbrot.hs | tail -n 1)
echo ""

# SUM
echo -n "sum"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/sum/sum.py | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/sum/sum.py | tail -n 1)
echo -n ","
# nothing for cpp
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/sum/sum.py | tail -n 1)
echo ""

# FLOAT
echo -n "float"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/float/float.py | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/float/float.py | tail -n 1)
echo -n ","
# nothing for cpp
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/float/float.py | tail -n 1)
echo ""

# GO
echo -n "go"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/go/go.py | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/go/go.py | tail -n 1)
echo -n ","
# nothing for cpp
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/go/go.hs | tail -n 1)
echo ""

# NBODY
echo -n "nbody"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/nbody/nbody.py 1000000 | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/nbody/nbody.py 1000000 | tail -n 1)
echo -n ","
echo -n $(${CPP} -std=c++17 -O3 ${BENCH_DIR}/nbody/nbody.cpp && ./a.out 1000000 | tail -n 1)
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/nbody/nbody.py 1000000 | tail -n 1)
echo ""

# CHAOS
echo -n "chaos"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/chaos/chaos.py /dev/null | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/chaos/chaos.py /dev/null | tail -n 1)
echo -n ","
# nothing for cpp
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/chaos/chaos.hs /dev/null | tail -n 1)
echo ""

# SPECTRAL_NORM
echo -n "spectral_norm"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/spectral_norm/spectral_norm.py | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/spectral_norm/spectral_norm.py | tail -n 1)
echo -n ","
# nothing for cpp
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/spectral_norm/spectral_norm.py | tail -n 1)
echo ""

# SET_PARTITION
echo -n "set_partition"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/set_partition/set_partition.py 15 | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/set_partition/set_partition.py 15 | tail -n 1)
echo -n ","
echo -n $(${CPP} -std=c++17 -O3 ${BENCH_DIR}/set_partition/set_partition.cpp && ./a.out 15 | tail -n 1)
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/set_partition/set_partition.py 15 | tail -n 1)
echo ""

# PRIMES
echo -n "primes"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/primes/primes.py 30000 | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/primes/primes.py 30000 | tail -n 1)
echo -n ","
# nothing for cpp
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/primes/primes.hs 30000 | tail -n 1)
echo ""

# BINARY_TREES
echo -n "binary_trees"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/binary_trees/binary_trees.py 20 | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/binary_trees/binary_trees.py 20 | tail -n 1)
echo -n ","
echo -n $(${CPP} -std=c++17 -O3 ${BENCH_DIR}/binary_trees/binary_trees.cpp && ./a.out 20 | tail -n 1)
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/binary_trees/binary_trees.hs 20 | tail -n 1)
echo ""

# FANNKUCH
echo -n "fannkuch"
echo -n ","
echo -n $(${PYTHON} ${BENCH_DIR}/fannkuch/fannkuch.py 11 | tail -n 1)
echo -n ","
echo -n $(${PYPY} ${BENCH_DIR}/fannkuch/fannkuch.py 11 | tail -n 1)
echo -n ","
# nothing for cpp
echo -n ","
echo -n $(${HERCULES} run -release ${BENCH_DIR}/fannkuch/fannkuch.hs 11 | tail -n 1)
echo ""

# WORD_COUNT
if [[ ! -z "${DATA_WORD_COUNT}" ]]; then
  echo -n "word_count"
  echo -n ","
  echo -n $(${PYTHON} ${BENCH_DIR}/word_count/word_count.py $DATA_WORD_COUNT | tail -n 1)
  echo -n ","
  echo -n $(${PYPY} ${BENCH_DIR}/word_count/word_count.py $DATA_WORD_COUNT | tail -n 1)
  echo -n ","
  echo -n $(${CPP} -std=c++17 -O3 ${BENCH_DIR}/word_count/word_count.cpp && ./a.out $DATA_WORD_COUNT | tail -n 1)
  echo -n ","
  echo -n $(${HERCULES} run -release ${BENCH_DIR}/word_count/word_count.py $DATA_WORD_COUNT | tail -n 1)
  echo ""
fi

# TAQ
if [[ ! -z "${DATA_TAQ}" ]]; then
  echo -n "taq"
  echo -n ","
  echo -n $(${PYTHON} ${BENCH_DIR}/taq/taq.py $DATA_TAQ | tail -n 1)
  echo -n ","
  echo -n $(${PYPY} ${BENCH_DIR}/taq/taq.py $DATA_TAQ | tail -n 1)
  echo -n ","
  echo -n $(${CPP} -std=c++17 -O3 ${BENCH_DIR}/taq/taq.cpp && ./a.out $DATA_TAQ | tail -n 1)
  echo -n ","
  echo -n $(${HERCULES} run -release ${BENCH_DIR}/taq/taq.py $DATA_TAQ | tail -n 1)
  echo ""
fi


