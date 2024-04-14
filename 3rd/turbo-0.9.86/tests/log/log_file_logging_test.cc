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

#include "turbo/testing/test.h"

#define SIMPLE_LOG "test_logs/simple_log"
#define ROTATING_LOG "test_logs/rotating_log"

TEST_CASE("simple_file_logger [simple_logger]]")
{
    prepare_logdir();
    turbo::tlog::filename_t filename = TLOG_FILENAME_T(SIMPLE_LOG);

    auto logger = turbo::tlog::create<turbo::tlog::sinks::basic_file_sink_mt>("logger", filename);
    logger->set_pattern("%v");

    logger->info("Test message {}", 1);
    logger->info("Test message {}", 2);

    logger->flush();
    require_message_count(SIMPLE_LOG, 2);
    using turbo::tlog::details::os::default_eol;
    REQUIRE(file_contents(SIMPLE_LOG) ==
            turbo::format("Test message 1{}Test message 2{}", default_eol, default_eol));
}

TEST_CASE("flush_on [flush_on]]")
{
    prepare_logdir();
    turbo::tlog::filename_t filename = TLOG_FILENAME_T(SIMPLE_LOG);

    auto logger = turbo::tlog::create<turbo::tlog::sinks::basic_file_sink_mt>("logger", filename);
    logger->set_pattern("%v");
    logger->set_level(turbo::tlog::level::trace);
    logger->flush_on(turbo::tlog::level::info);
    REQUIRE(count_lines(SIMPLE_LOG, true) == 0);
    logger->trace("Should not be flushed");
    REQUIRE(count_lines(SIMPLE_LOG, true) == 1);

    logger->info("Test message {}", 1);
    logger->info("Test message {}", 2);

    require_message_count(SIMPLE_LOG, 3);
    using turbo::tlog::details::os::default_eol;
    REQUIRE(file_contents(SIMPLE_LOG) ==
            turbo::format("Should not be flushed{}Test message 1{}Test message 2{}", default_eol,
                                         default_eol, default_eol));
}

TEST_CASE("rotating_file_logger1 [rotating_logger]]")
{
    prepare_logdir();
    size_t max_size = 1024 * 10;
    turbo::tlog::filename_t basename = TLOG_FILENAME_T(ROTATING_LOG);
    auto logger = turbo::tlog::rotating_logger_mt("logger", basename, max_size, 0);

    for (int i = 0; i < 10; ++i) {
        logger->info("Test message {}", i);
    }

    logger->flush();
    require_message_count(ROTATING_LOG, 10);
}

TEST_CASE("rotating_file_logger2 [rotating_logger]]")
{
    prepare_logdir();
    size_t max_size = 1024 * 10;
    turbo::tlog::filename_t basename = TLOG_FILENAME_T(ROTATING_LOG);

    {
        // make an initial logger to create the first output file
        auto logger = turbo::tlog::rotating_logger_mt("logger", basename, max_size, 2, true);
        for (int i = 0; i < 10; ++i) {
            logger->info("Test message {}", i);
        }
        // drop causes the logger destructor to be called, which is required so the
        // next logger can rename the first output file.
        turbo::tlog::drop(logger->name());
    }

    auto logger = turbo::tlog::rotating_logger_mt("logger", basename, max_size, 2, true);
    for (int i = 0; i < 10; ++i) {
        logger->info("Test message {}", i);
    }

    logger->flush();

    require_message_count(ROTATING_LOG, 10);

    for (int i = 0; i < 1000; i++) {

        logger->info("Test message {}", i);
    }

    logger->flush();
    REQUIRE(get_filesize(ROTATING_LOG) <= max_size);
    REQUIRE(get_filesize(ROTATING_LOG ".1") <= max_size);
}

// test that passing max_size=0 throws
TEST_CASE("rotating_file_logger3 [rotating_logger]]")
{
    prepare_logdir();
    size_t max_size = 0;
    turbo::tlog::filename_t basename = TLOG_FILENAME_T(ROTATING_LOG);
    REQUIRE_THROWS_AS(turbo::tlog::rotating_logger_mt("logger", basename, max_size, 0), turbo::tlog::tlog_ex);
}
