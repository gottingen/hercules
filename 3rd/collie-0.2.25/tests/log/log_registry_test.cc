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
static const char *const tested_logger_name = "null_logger";
static const char *const tested_logger_name2 = "null_logger2";

#ifndef TLOG_NO_EXCEPTIONS
TEST_CASE("register_drop [registry]")
{
    clog::drop_all();
    clog::create<clog::sinks::null_sink_mt>(tested_logger_name);
    REQUIRE(clog::get(tested_logger_name) != nullptr);
    // Throw if registering existing name
    REQUIRE_THROWS_AS(clog::create<clog::sinks::null_sink_mt>(tested_logger_name), clog::tlog_ex);
}

TEST_CASE("explicit register [registry]")
{
    clog::drop_all();
    auto logger = std::make_shared<clog::logger>(tested_logger_name, std::make_shared<clog::sinks::null_sink_st>());
    clog::register_logger(logger);
    REQUIRE(clog::get(tested_logger_name) != nullptr);
    // Throw if registering existing name
    REQUIRE_THROWS_AS(clog::create<clog::sinks::null_sink_mt>(tested_logger_name), clog::tlog_ex);
}
#endif

TEST_CASE("apply_all [registry]")
{
    clog::drop_all();
    auto logger = std::make_shared<clog::logger>(tested_logger_name, std::make_shared<clog::sinks::null_sink_st>());
    clog::register_logger(logger);
    auto logger2 = std::make_shared<clog::logger>(tested_logger_name2, std::make_shared<clog::sinks::null_sink_st>());
    clog::register_logger(logger2);

    int counter = 0;
    clog::apply_all([&counter](std::shared_ptr<clog::logger>) { counter++; });
    REQUIRE(counter == 2);

    counter = 0;
    clog::drop(tested_logger_name2);
    clog::apply_all([&counter](std::shared_ptr<clog::logger> l) {
        REQUIRE(l->name() == tested_logger_name);
        counter++;
    });
    REQUIRE(counter == 1);
}

TEST_CASE("drop [registry]")
{
    clog::drop_all();
    clog::create<clog::sinks::null_sink_mt>(tested_logger_name);
    clog::drop(tested_logger_name);
    REQUIRE_FALSE(clog::get(tested_logger_name));
}

TEST_CASE("drop-default [registry]")
{
    clog::set_default_logger(clog::null_logger_st(tested_logger_name));
    clog::drop(tested_logger_name);
    REQUIRE_FALSE(clog::default_logger());
    REQUIRE_FALSE(clog::get(tested_logger_name));
}

TEST_CASE("drop_all [registry]")
{
    clog::drop_all();
    clog::create<clog::sinks::null_sink_mt>(tested_logger_name);
    clog::create<clog::sinks::null_sink_mt>(tested_logger_name2);
    clog::drop_all();
    REQUIRE_FALSE(clog::get(tested_logger_name));
    REQUIRE_FALSE(clog::get(tested_logger_name2));
    REQUIRE_FALSE(clog::default_logger());
}

TEST_CASE("drop non existing [registry]")
{
    clog::drop_all();
    clog::create<clog::sinks::null_sink_mt>(tested_logger_name);
    clog::drop("some_name");
    REQUIRE_FALSE(clog::get("some_name"));
    REQUIRE(clog::get(tested_logger_name));
    clog::drop_all();
}

TEST_CASE("default logger [registry]")
{
    clog::drop_all();
    clog::set_default_logger(clog::null_logger_st(tested_logger_name));
    REQUIRE(clog::get(tested_logger_name) == clog::default_logger());
    clog::drop_all();
}

TEST_CASE("set_default_logger(nullptr) [registry]")
{
    clog::set_default_logger(nullptr);
    REQUIRE_FALSE(clog::default_logger());
}

TEST_CASE("disable automatic registration [registry]")
{
    // set some global parameters
    clog::level::level_enum log_level = clog::level::level_enum::warn;
    clog::set_level(log_level);
    // but disable automatic registration
    clog::set_automatic_registration(false);
    auto logger1 = clog::create<clog::sinks::daily_file_sink_st>(tested_logger_name, TLOG_FILENAME_T("filename"), 11, 59);
    auto logger2 = clog::create_async<clog::sinks::stdout_color_sink_mt>(tested_logger_name2);
    // loggers should not be part of the registry
    REQUIRE_FALSE(clog::get(tested_logger_name));
    REQUIRE_FALSE(clog::get(tested_logger_name2));
    // but make sure they are still initialized according to global defaults
    REQUIRE(logger1->level() == log_level);
    REQUIRE(logger2->level() == log_level);
    clog::set_level(clog::level::info);
    clog::set_automatic_registration(true);
}
