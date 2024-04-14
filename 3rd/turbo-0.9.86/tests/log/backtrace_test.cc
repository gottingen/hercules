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
#include "log_sink.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

TEST_CASE("bactrace1 [bactrace]")
{

    using turbo::tlog::sinks::test_sink_st;
    auto test_sink = std::make_shared<test_sink_st>();
    size_t backtrace_size = 5;

    turbo::tlog::logger logger("test-backtrace", test_sink);
    logger.set_pattern("%v");
    logger.enable_backtrace(backtrace_size);

    logger.info("info message");
    for (int i = 0; i < 100; i++)
        logger.debug("debug message {}", i);

    REQUIRE(test_sink->lines().size() == 1);
    REQUIRE(test_sink->lines()[0] == "info message");

    logger.dump_backtrace();
    REQUIRE(test_sink->lines().size() == backtrace_size + 3);
    REQUIRE(test_sink->lines()[1] == "****************** Backtrace Start ******************");
    REQUIRE(test_sink->lines()[2] == "debug message 95");
    REQUIRE(test_sink->lines()[3] == "debug message 96");
    REQUIRE(test_sink->lines()[4] == "debug message 97");
    REQUIRE(test_sink->lines()[5] == "debug message 98");
    REQUIRE(test_sink->lines()[6] == "debug message 99");
    REQUIRE(test_sink->lines()[7] == "****************** Backtrace End ********************");
}

TEST_CASE("bactrace-async [bactrace]")
{
    using turbo::tlog::sinks::test_sink_mt;
    auto test_sink = std::make_shared<test_sink_mt>();

    size_t backtrace_size = 5;

    turbo::tlog::init_thread_pool(120, 1);
    auto logger = std::make_shared<turbo::tlog::async_logger>("test-bactrace-async", test_sink, turbo::tlog::thread_pool());
    logger->set_pattern("%v");
    logger->enable_backtrace(backtrace_size);

    logger->info("info message");
    for (int i = 0; i < 100; i++)
        logger->debug("debug message {}", i);

    turbo::sleep_for(turbo::Duration::milliseconds(100));
    REQUIRE(test_sink->lines().size() == 1);
    REQUIRE(test_sink->lines()[0] == "info message");

    logger->dump_backtrace();
    turbo::sleep_for(turbo::Duration::milliseconds(100)); //  give time for the async dump to complete
    REQUIRE(test_sink->lines().size() == backtrace_size + 3);
    REQUIRE(test_sink->lines()[1] == "****************** Backtrace Start ******************");
    REQUIRE(test_sink->lines()[2] == "debug message 95");
    REQUIRE(test_sink->lines()[3] == "debug message 96");
    REQUIRE(test_sink->lines()[4] == "debug message 97");
    REQUIRE(test_sink->lines()[5] == "debug message 98");
    REQUIRE(test_sink->lines()[6] == "debug message 99");
    REQUIRE(test_sink->lines()[7] == "****************** Backtrace End ********************");
}
