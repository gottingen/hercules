#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


carbin_cc_benchmark(
        NAME
        raw_hash_set_benchmark
        SOURCES
        "raw_hash_set_benchmark.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        benchmark::benchmark
        benchmark::benchmark_main
)

carbin_cc_benchmark(
        NAME
        raw_hash_set_probe_benchmark
        SOURCES
        "raw_hash_set_probe_benchmark.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        benchmark::benchmark
        benchmark::benchmark_main
)

carbin_cc_benchmark(
        NAME
        layout_benchmark
        SOURCES
        "layout_benchmark.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        benchmark::benchmark
        benchmark::benchmark_main
)

carbin_cc_benchmark(
        NAME
        inlined_vector_benchmark
        SOURCES
        "inlined_vector_benchmark.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        benchmark::benchmark
        benchmark::benchmark_main
)

carbin_cc_benchmark(
        NAME
        fixed_array_benchmark
        SOURCES
        "fixed_array_benchmark.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        benchmark::benchmark
        benchmark::benchmark_main
)

carbin_cc_benchmark(
        NAME
        insert_benchmark
        SOURCES
        "Insert.cc"
        "main.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)
