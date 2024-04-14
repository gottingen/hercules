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

#include <cstdlib>
#include <turbo/log/cfg/env.h>
#include <turbo/log/cfg/argv.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

using turbo::tlog::cfg::load_argv_levels;
using turbo::tlog::cfg::load_env_levels;
using turbo::tlog::sinks::test_sink_st;

TEST_CASE("env [cfg]")
{
    turbo::tlog::drop("l1");
    auto l1 = turbo::tlog::create<test_sink_st>("l1");
#ifdef CATCH_PLATFORM_WINDOWS
    _putenv_s("TLOG_LEVEL", "l1=warn");
#else
    ::setenv("TLOG_LEVEL", "l1=warn", 1);
#endif
    load_env_levels();
    REQUIRE_EQ(l1->level(), turbo::tlog::level::warn);
    turbo::tlog::set_default_logger(turbo::tlog::create<test_sink_st>("cfg-default"));
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::info);
}

TEST_CASE("argv1 [cfg]")
{
    turbo::tlog::drop("l1");
    const char *argv[] = {"ignore", "TLOG_LEVEL=l1=warn"};
    load_argv_levels(2, argv);
    auto l1 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l1");
    REQUIRE_EQ(l1->level(), turbo::tlog::level::warn);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::info);
}

TEST_CASE("argv2 [cfg]")
{
    turbo::tlog::drop("l1");
    const char *argv[] = {"ignore", "TLOG_LEVEL=l1=warn,trace"};
    load_argv_levels(2, argv);
    auto l1 = turbo::tlog::create<test_sink_st>("l1");
    REQUIRE_EQ(l1->level(), turbo::tlog::level::warn);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::trace);
}

TEST_CASE("argv3 [cfg]")
{
    turbo::tlog::set_level(turbo::tlog::level::trace);

    turbo::tlog::drop("l1");
    const char *argv[] = {"ignore", "TLOG_LEVEL=junk_name=warn"};
    load_argv_levels(2, argv);
    auto l1 = turbo::tlog::create<test_sink_st>("l1");
    REQUIRE_EQ(l1->level(), turbo::tlog::level::trace);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::trace);
}

TEST_CASE("argv4 [cfg]")
{
    turbo::tlog::set_level(turbo::tlog::level::info);
    turbo::tlog::drop("l1");
    const char *argv[] = {"ignore", "TLOG_LEVEL=junk"};
    load_argv_levels(2, argv);
    auto l1 = turbo::tlog::create<test_sink_st>("l1");
    REQUIRE_EQ(l1->level(), turbo::tlog::level::info);
}

TEST_CASE("argv5 [cfg]")
{
    turbo::tlog::set_level(turbo::tlog::level::info);
    turbo::tlog::drop("l1");
    const char *argv[] = {"ignore", "ignore", "TLOG_LEVEL=l1=warn,trace"};
    load_argv_levels(3, argv);
    auto l1 = turbo::tlog::create<test_sink_st>("l1");
    REQUIRE_EQ(l1->level(), turbo::tlog::level::warn);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::trace);
    turbo::tlog::set_level(turbo::tlog::level::info);
}

TEST_CASE("argv6 [cfg]")
{
    turbo::tlog::set_level(turbo::tlog::level::err);
    const char *argv[] = {""};
    load_argv_levels(1, argv);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::err);
    turbo::tlog::set_level(turbo::tlog::level::info);
}

TEST_CASE("argv7 [cfg]")
{
    turbo::tlog::set_level(turbo::tlog::level::err);
    const char *argv[] = {""};
    load_argv_levels(0, argv);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::err);
    turbo::tlog::set_level(turbo::tlog::level::info);
}

TEST_CASE("level-not-set-test1 [cfg]")
{
    turbo::tlog::drop("l1");
    const char *argv[] = {"ignore", ""};
    load_argv_levels(2, argv);
    auto l1 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l1");
    l1->set_level(turbo::tlog::level::trace);
    REQUIRE_EQ(l1->level(), turbo::tlog::level::trace);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::info);
}

TEST_CASE("level-not-set-test2 [cfg]")
{
    turbo::tlog::drop("l1");
    turbo::tlog::drop("l2");
    const char *argv[] = {"ignore", "TLOG_LEVEL=l1=trace"};

    auto l1 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l1");
    l1->set_level(turbo::tlog::level::warn);
    auto l2 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l2");
    l2->set_level(turbo::tlog::level::warn);

    load_argv_levels(2, argv);

    REQUIRE_EQ(l1->level(), turbo::tlog::level::trace);
    REQUIRE_EQ(l2->level(), turbo::tlog::level::warn);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::info);
}

TEST_CASE("level-not-set-test3 [cfg]")
{
    turbo::tlog::drop("l1");
    turbo::tlog::drop("l2");
    const char *argv[] = {"ignore", "TLOG_LEVEL=l1=trace"};

    load_argv_levels(2, argv);

    auto l1 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l1");
    auto l2 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l2");

    REQUIRE_EQ(l1->level(), turbo::tlog::level::trace);
    REQUIRE_EQ(l2->level(), turbo::tlog::level::info);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::info);
}

TEST_CASE("level-not-set-test4 [cfg]")
{
    turbo::tlog::drop("l1");
    turbo::tlog::drop("l2");
    const char *argv[] = {"ignore", "TLOG_LEVEL=l1=trace,warn"};

    load_argv_levels(2, argv);

    auto l1 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l1");
    auto l2 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l2");

    REQUIRE_EQ(l1->level(), turbo::tlog::level::trace);
    REQUIRE_EQ(l2->level(), turbo::tlog::level::warn);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::warn);
}

TEST_CASE("level-not-set-test5 [cfg]")
{
    turbo::tlog::drop("l1");
    turbo::tlog::drop("l2");
    const char *argv[] = {"ignore", "TLOG_LEVEL=l1=junk,warn"};

    load_argv_levels(2, argv);

    auto l1 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l1");
    auto l2 = turbo::tlog::create<turbo::tlog::sinks::test_sink_st>("l2");

    REQUIRE_EQ(l1->level(), turbo::tlog::level::warn);
    REQUIRE_EQ(l2->level(), turbo::tlog::level::warn);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::warn);
}

TEST_CASE("restore-to-default [cfg]")
{
    turbo::tlog::drop("l1");
    turbo::tlog::drop("l2");
    const char *argv[] = {"ignore", "TLOG_LEVEL=info"};
    load_argv_levels(2, argv);
    REQUIRE_EQ(turbo::tlog::default_logger()->level(), turbo::tlog::level::info);
}
