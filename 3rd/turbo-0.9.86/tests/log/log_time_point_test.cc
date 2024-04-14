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
#include "turbo/log/async.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

TEST_CASE("time_point1 [time_point log_msg]")
{
    std::shared_ptr<turbo::tlog::sinks::test_sink_st> test_sink(new turbo::tlog::sinks::test_sink_st);
    turbo::tlog::logger logger("test-time_point", test_sink);

    turbo::tlog::source_loc source{};
    turbo::Time tp = turbo::Time::time_now();
    test_sink->set_pattern("%T.%F"); // interested in the time_point

    // all the following should have the same time
    test_sink->set_delay(std::chrono::milliseconds(10));
    for (int i = 0; i < 5; i++)
    {
        turbo::tlog::details::log_msg msg{tp, source, "test_logger", turbo::tlog::level::info, "message"};
        test_sink->log(msg);
    }

    logger.log(tp, source, turbo::tlog::level::info, "formatted message");
    logger.log(tp, source, turbo::tlog::level::info, "formatted message");
    logger.log(tp, source, turbo::tlog::level::info, "formatted message");
    logger.log(tp, source, turbo::tlog::level::info, "formatted message");
    logger.log(source, turbo::tlog::level::info, "formatted message"); // last line has different time_point

    // now the real test... that the times are the same.
    std::vector<std::string> lines = test_sink->lines();
    REQUIRE_EQ(lines[0] , lines[1]);
    REQUIRE_EQ(lines[2] , lines[3]);
    REQUIRE_EQ(lines[4] , lines[5]);
    REQUIRE_EQ(lines[6] , lines[7]);
    REQUIRE_NE(lines[8] , lines[9]);
    turbo::tlog::drop_all();
}
