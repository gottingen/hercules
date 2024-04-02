

find_path(BENCHMARK_INCLUDE_PATH NAMES benchmark/benchmark.h)
find_library(BENCHMARK_LIB NAMES libbenchmark.a benchmark)
find_library(BENCHMARK_MAIN_LIB NAMES libbenchmark_main.a benchmark_main)
if ((NOT BENCHMARK_INCLUDE_PATH) OR (NOT BENCHMARK_LIB) OR (NOT BENCHMARK_MAIN_LIB))
    carbin_error("Fail to find benchmark")
endif ()
include_directories(${BENCHMARK_INCLUDE_PATH})