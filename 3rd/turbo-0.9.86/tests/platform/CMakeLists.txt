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



# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        atomic_hook_test_helper
        SOURCES
        "atomic_hook_test_helper.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)

carbin_cc_test(
        NAME
        atomic_hook_test
        SOURCES
        "atomic_hook_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::atomic_hook_test_helper
        turbo::turbo
)


carbin_cc_test(
        NAME
        errno_saver_test
        SOURCES
        "errno_saver_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)

carbin_cc_test(
        NAME
        config_test
        SOURCES
        "config_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)


carbin_cc_test(
        NAME
        low_level_alloc_test
        SOURCES
        "low_level_alloc_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
        Threads::Threads
)

carbin_cc_test(
        NAME
        prefetch_test
        SOURCES
        "prefetch_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)

carbin_cc_test(
        NAME
        optimization_test
        SOURCES
        "optimization_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)

carbin_cc_test(
        NAME
        unique_small_name_test
        SOURCES
        "unique_small_name_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)
