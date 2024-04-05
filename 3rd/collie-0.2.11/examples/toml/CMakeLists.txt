#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

add_custom_target(run_examples COMMENT "Running all examples")

function(add_example name)
  cmake_parse_arguments(PARSE_ARGV 1 "" "" "" ARGS)
  carbin_cc_binary(
          NAME toml_${name}
          SOURCES ${name}.cc
          COPTS ${USER_CXX_FLAGS}
  )
endfunction()

add_example(error_printer)
add_example(parse_benchmark)
add_example(simple_parser)
add_example(toml_generator)
add_example(toml_merger)
add_example(toml_to_json_transcoder)
