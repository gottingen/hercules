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

carbin_cc_test(
        NAME
        iobuf_test
        SOURCES
        "iobuf_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-fno-access-control"
        DEPS
        turbo::turbo
)

carbin_cc_test(
        NAME
        env_test
        SOURCES
        "env_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)

carbin_cc_test(
        NAME
        endpoint_test
        SOURCES
        "endpoint_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-fno-access-control"
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)

carbin_cc_test(
        NAME
        processor_test
        SOURCES
        "processor_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)

carbin_cc_test(
        NAME
        sysinfo_test
        SOURCES
        "sysinfo_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)

carbin_cc_test(
        NAME
        atexit_test
        SOURCES
        "atexit_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)
