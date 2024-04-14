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

carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        hash_testing
        HEADERS
        "hash_testing.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${CARBIN_DEPS_LINK}
)

carbin_cc_test(
        NAME
        hash_test
        SOURCES
        "hash_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::hash_testing
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
#
# Note: Even though external code should not depend on this target
# directly, it must be marked PUBLIC since it is a dependency of
# hash_testing.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        spy_hash_state
        HEADERS
        "spy_hash_state.h"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        DEPS
        turbo::hash
        turbo::strings
        turbo::str_format
        ${CARBIN_DEPS_LINK}
)


carbin_cc_test(
        NAME
        city_test
        SOURCES
        "city_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${CARBIN_DEPS_LINK}
)


carbin_cc_test(
        NAME
        bytes_hash_test
        SOURCES
        "bytes_hash_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${CARBIN_DEPS_LINK}
)


carbin_cc_test(
        NAME
        engine_test
        SOURCES
        "engine_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${CARBIN_DEPS_LINK}
)
