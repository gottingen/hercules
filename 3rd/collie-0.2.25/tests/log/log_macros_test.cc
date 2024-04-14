// Copyright 2023 The Turbo Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "includes.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "collie/testing/test.h"

#if TLOG_ACTIVE_LEVEL != TLOG_LEVEL_DEBUG
#    error "Invalid TLOG_ACTIVE_LEVEL in test. Should be TLOG_LEVEL_DEBUG"
#endif

#define TEST_FILENAME "test_logs/simple_log"

TEST_CASE("debug and trace w/o format string [macros]]")
{

    prepare_logdir();
    clog::filename_t filename = TLOG_FILENAME_T(TEST_FILENAME);

    auto logger = clog::create<clog::sinks::basic_file_sink_mt>("logger", filename);
    logger->set_pattern("%v");
    logger->set_level(clog::level::trace);

    TLOG_LOGGER_TRACE(logger, "Test message 1");
    TLOG_LOGGER_DEBUG(logger, "Test message 2");
    logger->flush();

    using clog::details::os::default_eol;
    REQUIRE(ends_with(file_contents(TEST_FILENAME), turbo::format("Test message 2{}", default_eol)));
    REQUIRE(count_lines(TEST_FILENAME) == 1);

    auto orig_default_logger = clog::default_logger();
    clog::set_default_logger(logger);

    TLOG_TRACE("Test message 3");
    TLOG_DEBUG("Test message {}", 4);
    logger->flush();

    require_message_count(TEST_FILENAME, 2);
    REQUIRE(ends_with(file_contents(TEST_FILENAME), turbo::format("Test message 4{}", default_eol)));
    clog::set_default_logger(std::move(orig_default_logger));
}

TEST_CASE("disable param evaluation [macros]")
{
    TLOG_TRACE("Test message {}", throw std::runtime_error("Should not be evaluated"));
}

TEST_CASE("pass logger pointer [macros]")
{
    auto logger = clog::create<clog::sinks::null_sink_mt>("refmacro");
    auto &ref = *logger;
    TLOG_LOGGER_TRACE(&ref, "Test message 1");
    TLOG_LOGGER_DEBUG(&ref, "Test message 2");
}
