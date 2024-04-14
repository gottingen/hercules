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
        log_utils
        SOURCES
        "utils.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        DEPS
        turbo::turbo
)

carbin_cc_test(
        NAME
        daily_logger_test
        SOURCES
        "daily_logger_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)
carbin_cc_test(
        NAME
        log_async_test
        SOURCES
        "async_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)
carbin_cc_test(
        NAME
        backtrace_test
        SOURCES
        "backtrace_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_cfg_test
        SOURCES
        "log_cfg_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_dup_filter_test
        SOURCES
        "log_dup_filter_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_file_logging_test
        SOURCES
        "log_file_logging_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)


carbin_cc_test(
        NAME
        log_fmt_helper_test
        SOURCES
        "log_fmt_helper_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_macros_test
        SOURCES
        "log_macros_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_misc_test
        SOURCES
        "log_misc_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_mpmc_q_test
        SOURCES
        "log_mpmc_q_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_pattern_formatter_test
        SOURCES
        "log_pattern_formatter_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_registry_test
        SOURCES
        "log_registry_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_stdout_api_test
        SOURCES
        "log_stdout_api_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)

carbin_cc_test(
        NAME
        log_time_point_test
        SOURCES
        "log_time_point_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::log_utils
)