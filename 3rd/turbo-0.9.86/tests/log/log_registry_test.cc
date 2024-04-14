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
static const char *const tested_logger_name = "null_logger";
static const char *const tested_logger_name2 = "null_logger2";

#ifndef TLOG_NO_EXCEPTIONS
TEST_CASE("register_drop [registry]")
{
    turbo::tlog::drop_all();
    turbo::tlog::create<turbo::tlog::sinks::null_sink_mt>(tested_logger_name);
    REQUIRE(turbo::tlog::get(tested_logger_name) != nullptr);
    // Throw if registering existing name
    REQUIRE_THROWS_AS(turbo::tlog::create<turbo::tlog::sinks::null_sink_mt>(tested_logger_name), turbo::tlog::tlog_ex);
}

TEST_CASE("explicit register [registry]")
{
    turbo::tlog::drop_all();
    auto logger = std::make_shared<turbo::tlog::logger>(tested_logger_name, std::make_shared<turbo::tlog::sinks::null_sink_st>());
    turbo::tlog::register_logger(logger);
    REQUIRE(turbo::tlog::get(tested_logger_name) != nullptr);
    // Throw if registering existing name
    REQUIRE_THROWS_AS(turbo::tlog::create<turbo::tlog::sinks::null_sink_mt>(tested_logger_name), turbo::tlog::tlog_ex);
}
#endif

TEST_CASE("apply_all [registry]")
{
    turbo::tlog::drop_all();
    auto logger = std::make_shared<turbo::tlog::logger>(tested_logger_name, std::make_shared<turbo::tlog::sinks::null_sink_st>());
    turbo::tlog::register_logger(logger);
    auto logger2 = std::make_shared<turbo::tlog::logger>(tested_logger_name2, std::make_shared<turbo::tlog::sinks::null_sink_st>());
    turbo::tlog::register_logger(logger2);

    int counter = 0;
    turbo::tlog::apply_all([&counter](std::shared_ptr<turbo::tlog::logger>) { counter++; });
    REQUIRE(counter == 2);

    counter = 0;
    turbo::tlog::drop(tested_logger_name2);
    turbo::tlog::apply_all([&counter](std::shared_ptr<turbo::tlog::logger> l) {
        REQUIRE(l->name() == tested_logger_name);
        counter++;
    });
    REQUIRE(counter == 1);
}

TEST_CASE("drop [registry]")
{
    turbo::tlog::drop_all();
    turbo::tlog::create<turbo::tlog::sinks::null_sink_mt>(tested_logger_name);
    turbo::tlog::drop(tested_logger_name);
    REQUIRE_FALSE(turbo::tlog::get(tested_logger_name));
}

TEST_CASE("drop-default [registry]")
{
    turbo::tlog::set_default_logger(turbo::tlog::null_logger_st(tested_logger_name));
    turbo::tlog::drop(tested_logger_name);
    REQUIRE_FALSE(turbo::tlog::default_logger());
    REQUIRE_FALSE(turbo::tlog::get(tested_logger_name));
}

TEST_CASE("drop_all [registry]")
{
    turbo::tlog::drop_all();
    turbo::tlog::create<turbo::tlog::sinks::null_sink_mt>(tested_logger_name);
    turbo::tlog::create<turbo::tlog::sinks::null_sink_mt>(tested_logger_name2);
    turbo::tlog::drop_all();
    REQUIRE_FALSE(turbo::tlog::get(tested_logger_name));
    REQUIRE_FALSE(turbo::tlog::get(tested_logger_name2));
    REQUIRE_FALSE(turbo::tlog::default_logger());
}

TEST_CASE("drop non existing [registry]")
{
    turbo::tlog::drop_all();
    turbo::tlog::create<turbo::tlog::sinks::null_sink_mt>(tested_logger_name);
    turbo::tlog::drop("some_name");
    REQUIRE_FALSE(turbo::tlog::get("some_name"));
    REQUIRE(turbo::tlog::get(tested_logger_name));
    turbo::tlog::drop_all();
}

TEST_CASE("default logger [registry]")
{
    turbo::tlog::drop_all();
    turbo::tlog::set_default_logger(turbo::tlog::null_logger_st(tested_logger_name));
    REQUIRE(turbo::tlog::get(tested_logger_name) == turbo::tlog::default_logger());
    turbo::tlog::drop_all();
}

TEST_CASE("set_default_logger(nullptr) [registry]")
{
    turbo::tlog::set_default_logger(nullptr);
    REQUIRE_FALSE(turbo::tlog::default_logger());
}

TEST_CASE("disable automatic registration [registry]")
{
    // set some global parameters
    turbo::tlog::level::level_enum log_level = turbo::tlog::level::level_enum::warn;
    turbo::tlog::set_level(log_level);
    // but disable automatic registration
    turbo::tlog::set_automatic_registration(false);
    auto logger1 = turbo::tlog::create<turbo::tlog::sinks::daily_file_sink_st>(tested_logger_name, TLOG_FILENAME_T("filename"), 11, 59);
    auto logger2 = turbo::tlog::create_async<turbo::tlog::sinks::stdout_color_sink_mt>(tested_logger_name2);
    // loggers should not be part of the registry
    REQUIRE_FALSE(turbo::tlog::get(tested_logger_name));
    REQUIRE_FALSE(turbo::tlog::get(tested_logger_name2));
    // but make sure they are still initialized according to global defaults
    REQUIRE(logger1->level() == log_level);
    REQUIRE(logger2->level() == log_level);
    turbo::tlog::set_level(turbo::tlog::level::info);
    turbo::tlog::set_automatic_registration(true);
}
