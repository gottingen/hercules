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
#include "turbo/log/sinks/dup_filter_sink.h"
#include "log_sink.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

TEST_CASE("dup_filter_test1 [dup_filter_sink]")
{
    using turbo::tlog::sinks::dup_filter_sink_st;
    using turbo::tlog::sinks::test_sink_mt;

    dup_filter_sink_st dup_sink{turbo::Duration::seconds(5)};
    auto test_sink = std::make_shared<test_sink_mt>();
    dup_sink.add_sink(test_sink);

    for (int i = 0; i < 10; i++)
    {
        dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message1"});
    }

    REQUIRE_EQ(test_sink->msg_counter() , 1);
}

TEST_CASE("dup_filter_test2 [dup_filter_sink]")
{
    using turbo::tlog::sinks::dup_filter_sink_st;
    using turbo::tlog::sinks::test_sink_mt;

    dup_filter_sink_st dup_sink{turbo::Duration::seconds(0)};
    auto test_sink = std::make_shared<test_sink_mt>();
    dup_sink.add_sink(test_sink);

    for (int i = 0; i < 10; i++)
    {
        dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message1"});
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    REQUIRE_EQ(test_sink->msg_counter() , 10);
}

TEST_CASE("dup_filter_test3 [dup_filter_sink]")
{
    using turbo::tlog::sinks::dup_filter_sink_st;
    using turbo::tlog::sinks::test_sink_mt;

    dup_filter_sink_st dup_sink(turbo::Duration::seconds(1));
    auto test_sink = std::make_shared<test_sink_mt>();
    dup_sink.add_sink(test_sink);

    for (int i = 0; i < 10; i++)
    {
        dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message1"});
        dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message2"});
    }

    REQUIRE_EQ(test_sink->msg_counter() , 20);
}

TEST_CASE("dup_filter_test4 [dup_filter_sink]")
{
    using turbo::tlog::sinks::dup_filter_sink_mt;
    using turbo::tlog::sinks::test_sink_mt;

    dup_filter_sink_mt dup_sink{turbo::Duration::milliseconds(10)};
    auto test_sink = std::make_shared<test_sink_mt>();
    dup_sink.add_sink(test_sink);

    dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message"});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message"});
    REQUIRE_EQ(test_sink->msg_counter() , 2);
}

TEST_CASE("dup_filter_test5 [dup_filter_sink]")
{
    using turbo::tlog::sinks::dup_filter_sink_mt;
    using turbo::tlog::sinks::test_sink_mt;

    dup_filter_sink_mt dup_sink{turbo::Duration::seconds(5)};
    auto test_sink = std::make_shared<test_sink_mt>();
    test_sink->set_pattern("%v");
    dup_sink.add_sink(test_sink);

    dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message1"});
    dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message1"});
    dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message1"});
    dup_sink.log(turbo::tlog::details::log_msg{"test", turbo::tlog::level::info, "message2"});

    REQUIRE_EQ(test_sink->msg_counter() , 3); // skip 2 messages but log the "skipped.." message before message2
    REQUIRE_EQ(test_sink->lines()[1] , "Skipped 2 duplicate messages..");
}
