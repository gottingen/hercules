#
# Copyright 2023 The Carbin Authors.
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

include(CMakeParseArguments)
include(carbin_config_cxx_opts)
include(carbin_install_dirs)
include(carbin_print)

function(carbin_cc_benchmark)

    set(list_args
            DEPS
            SOURCES
            DEFINITIONS
            COPTS
            )

    cmake_parse_arguments(
            CARBIN_CC_BENCHMARK
            ""
            "NAME"
            "DEPS;SOURCES;DEFINITIONS;COPTS"
            ${ARGN}
    )

    if (NOT CARBIN_BUILD_BENCHMARK)
        return()
    endif ()

    carbin_raw("-----------------------------------")
    carbin_print_label("Building Test" "${CARBIN_CC_BENCHMARK_NAME}")
    carbin_raw("-----------------------------------")
    if (VERBOSE_CARBIN_BUILD)
        carbin_print_list_label("Sources" CARBIN_CC_BENCHMARK_SOURCES)
        carbin_print_list_label("DEPS" CARBIN_CC_BENCHMARK_DEPS)
        carbin_print_list_label("COPTS" CARBIN_CC_BENCHMARK_COPTS)
        carbin_print_list_label("DEFINITIONS" CARBIN_CC_BENCHMARK_DEFINITIONS)
        message("-----------------------------------")
    endif ()

    set(testcase ${CARBIN_CC_BENCHMARK_NAME})

    add_executable(${testcase} ${CARBIN_CC_BENCHMARK_SOURCES})

    target_compile_options(${testcase} PRIVATE ${CARBIN_CC_BENCHMARK_COPTS})
    target_link_libraries(${testcase} PRIVATE ${CARBIN_CC_BENCHMARK_DEPS})

    target_compile_definitions(${testcase}
            PUBLIC
            ${CARBIN_CC_BENCHMARK_DEFINITIONS}
            )

    target_include_directories(${testcase} ${CARBIN_INTERNAL_INCLUDE_WARNING_GUARD}
            PUBLIC
            "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
            "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
            )
    if (NOT CARBIN_CC_BENCHMARK_COMMAND)
        set(CARBIN_CC_BENCHMARK_COMMAND ${testcase})
    endif ()
    add_test(NAME ${testcase}
            COMMAND ${CARBIN_CC_BENCHMARK_COMMAND}
            )

endfunction()
