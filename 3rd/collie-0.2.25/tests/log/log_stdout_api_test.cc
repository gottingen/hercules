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
#include "turbo/log/sinks/stdout_sinks.h"
#include "turbo/log/sinks/stdout_color_sinks.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "collie/testing/test.h"

TEST_CASE("stdout_st [stdout]")
{
    auto l = clog::stdout_logger_st("test");
    l->set_pattern("%+");
    l->set_level(clog::level::trace);
    l->trace("Test stdout_st");
    clog::drop_all();
}

TEST_CASE("stdout_mt [stdout]")
{
    auto l = clog::stdout_logger_mt("test");
    l->set_pattern("%+");
    l->set_level(clog::level::debug);
    l->debug("Test stdout_mt");
    clog::drop_all();
}

TEST_CASE("stderr_st [stderr]")
{
    auto l = clog::stderr_logger_st("test");
    l->set_pattern("%+");
    l->info("Test stderr_st");
    clog::drop_all();
}

TEST_CASE("stderr_mt [stderr]")
{
    auto l = clog::stderr_logger_mt("test");
    l->set_pattern("%+");
    l->info("Test stderr_mt");
    l->warn("Test stderr_mt");
    l->error("Test stderr_mt");
    l->critical("Test stderr_mt");
    clog::drop_all();
}

// color loggers
TEST_CASE("stdout_color_st [stdout]")
{
    auto l = clog::stdout_color_st("test");
    l->set_pattern("%+");
    l->info("Test stdout_color_st");
    clog::drop_all();
}

TEST_CASE("stdout_color_mt [stdout]")
{
    auto l = clog::stdout_color_mt("test");
    l->set_pattern("%+");
    l->set_level(clog::level::trace);
    l->trace("Test stdout_color_mt");
    clog::drop_all();
}

TEST_CASE("stderr_color_st [stderr]")
{
    auto l = clog::stderr_color_st("test");
    l->set_pattern("%+");
    l->set_level(clog::level::debug);
    l->debug("Test stderr_color_st");
    clog::drop_all();
}

TEST_CASE("stderr_color_mt [stderr]")
{
    auto l = clog::stderr_color_mt("test");
    l->set_pattern("%+");
    l->info("Test stderr_color_mt");
    l->warn("Test stderr_color_mt");
    l->error("Test stderr_color_mt");
    l->critical("Test stderr_color_mt");
    clog::drop_all();
}

#ifdef TLOG_WCHAR_TO_UTF8_SUPPORT

TEST_CASE("wchar_api [stdout]")
{
    auto l = clog::stdout_logger_st("wchar_logger");
    l->set_pattern("%+");
    l->set_level(clog::level::trace);
    l->trace(L"Test wchar_api");
    l->trace(L"Test wchar_api {}", L"param");
    l->trace(L"Test wchar_api {}", 1);
    l->trace(L"Test wchar_api {}", std::wstring{L"wstring param"});
    l->trace(std::wstring{L"Test wchar_api wstring"});
    TLOG_LOGGER_DEBUG(l, L"Test TLOG_LOGGER_DEBUG {}", L"param");
    clog::drop_all();
}

#endif
