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
        type_traits_test
        SOURCES
        "type_traits_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)


carbin_cc_test(
        NAME
        utility_test
        SOURCES
        "utility_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)


carbin_cc_test(
        NAME
        span_test
        SOURCES
        "span_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::exception_testing
        turbo::hash_testing
        turbo::turbo
)

carbin_cc_test(
        NAME
        span_test_noexceptions
        SOURCES
        "span_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::exception_testing
        turbo::hash_testing
        turbo::turbo
)

carbin_cc_test(
        NAME
        compare_test
        SOURCES
        "compare_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)



carbin_cc_test(
        NAME
        algorithm_test
        SOURCES
        "algorithm_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)


carbin_cc_test(
        NAME
        container_test
        SOURCES
        "container_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)


carbin_cc_test(
        NAME
        bind_front_test
        SOURCES
        "bind_front_test.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        DEPS
        turbo::turbo
)

carbin_cc_test(
        NAME
        function_ref_test
        SOURCES
        "function_ref_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
        turbo::test_instance_tracker
)

carbin_cc_test(
        NAME
        unique_generic_test
        SOURCES
        "unique_generic_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
        turbo::test_instance_tracker
)
