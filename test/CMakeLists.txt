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

include(FetchContent)
FetchContent_Declare(
        googletest
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/googletest-release-1.12.1/googletest-release-1.12.1"
        #URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_library(testing OBJECT testing.cc types.cc)
target_include_directories(testing PRIVATE hir
        "${gc_SOURCE_DIR}/include" "${googletest_SOURCE_DIR}/googletest/include")
target_compile_definitions(testing
        PRIVATE TEST_DIR="${PROJECT_SOURCE_DIR}/test")
set(TEST_FILES
        hercules_parse_test
        hercules_transform_test
        hercules_core_test
        hercules_numerics_test
        hercules_stdlib_test)

foreach (TEST_FILE ${TEST_FILES})
    add_executable(${TEST_FILE} ${TEST_FILE}.cc)
    target_include_directories(${TEST_FILE} PRIVATE hir
            "${gc_SOURCE_DIR}/include")
    target_link_libraries(${TEST_FILE} herculesc herculesrt
            $<TARGET_OBJECTS:testing> gtest_main)
    target_compile_definitions(${TEST_FILE}
            PRIVATE TEST_DIR="${PROJECT_SOURCE_DIR}/test")
endforeach ()

#[[
set(HERCULES_TEST_CPPFILES
        hercules_test.cc
        types.cc)
add_executable(hercules_test ${HERCULES_TEST_CPPFILES})
target_include_directories(hercules_test PRIVATE hir
        "${gc_SOURCE_DIR}/include")
target_link_libraries(hercules_test herculesc herculesrt gtest_main)
target_compile_definitions(hercules_test
        PRIVATE TEST_DIR="${PROJECT_SOURCE_DIR}/test")
]]

add_subdirectory(ast)

set(HIR_TEST_CPPFILES
        hir_test.cc
        hir/analyze/dominator.cc
        hir/analyze/reaching.cc
        hir/base.cc
        hir/constant.cc
        hir/flow.cc
        hir/func.cc
        hir/instr.cc
        hir/module.cc
        hir/transform/manager.cc
        hir/types/types.cc
        hir/util/matching.cc
        hir/value.cc
        hir/var.cc
        types.cc)
add_executable(hir_test ${HIR_TEST_CPPFILES})
target_include_directories(hir_test PRIVATE hir
        "${gc_SOURCE_DIR}/include")
target_link_libraries(hir_test herculesc herculesrt gtest_main)
target_compile_definitions(hir_test
        PRIVATE TEST_DIR="${PROJECT_SOURCE_DIR}/test")
