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

#[[carbin_cc_test(
        NAME custom_comparator_test
        SOURCES custom_comparator_test.cc
        DEPS collie::collie ${CARBIN_DEPS_LINK} ${GTEST_LIB} ${GTEST_MAIN_LIB}
        COPTS ${USER_CXX_FLAGS}
)]]
#[[
carbin_cc_test(
        NAME foo_test
        SOURCES foo_test.cc
        DEPS changeme::foo ${CARBIN_DEPS_LINK} gtest gtest_main
        COPTS ${USER_CXX_FLAGS}
)

carbin_cc_test(
        NAME foo_doctest
        SOURCES foo_doctest.cc
        DEPS changeme::foo ${CARBIN_DEPS_LINK} doctest
        COPTS ${USER_CXX_FLAGS}
)
]]
add_subdirectory(base)
add_subdirectory(container)
add_subdirectory(meta)
add_subdirectory(strings)
add_subdirectory(simd)
add_subdirectory(tc)
add_subdirectory(taskflow)