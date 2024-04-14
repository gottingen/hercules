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
#include "turbo/log/fmt/bin_to_hex.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

template<class T>
std::string log_info(const T &what, turbo::tlog::level::level_enum logger_level = turbo::tlog::level::info)
{

    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);

    turbo::tlog::logger oss_logger("oss", oss_sink);
    oss_logger.set_level(logger_level);
    oss_logger.set_pattern("%v");
    oss_logger.info(what);

    return oss.str().substr(0, oss.str().length() - strlen(turbo::tlog::details::os::default_eol));
}

TEST_CASE("basic_logging  [basic_logging]")
{
    // const char
    REQUIRE(log_info("Hello") == "Hello");
    REQUIRE(log_info("").empty());

    // std::string
    REQUIRE(log_info(std::string("Hello")) == "Hello");
    REQUIRE(log_info(std::string()).empty());

    // Numbers
    REQUIRE(log_info(5) == "5");
    REQUIRE(log_info(5.6) == "5.6");

    // User defined class
    // REQUIRE(log_info(some_logged_class("some_val")) == "some_val");
}

TEST_CASE("log_levels [log_levels]")
{
    REQUIRE(log_info("Hello", turbo::tlog::level::err).empty());
    REQUIRE(log_info("Hello", turbo::tlog::level::critical).empty());
    REQUIRE(log_info("Hello", turbo::tlog::level::info) == "Hello");
    REQUIRE(log_info("Hello", turbo::tlog::level::debug) == "Hello");
    REQUIRE(log_info("Hello", turbo::tlog::level::trace) == "Hello");
}

TEST_CASE("level_to_string_view [convert_to_string_view")
{
    REQUIRE(turbo::tlog::level::to_string_view(turbo::tlog::level::trace) == "trace");
    REQUIRE(turbo::tlog::level::to_string_view(turbo::tlog::level::debug) == "debug");
    REQUIRE(turbo::tlog::level::to_string_view(turbo::tlog::level::info) == "info");
    REQUIRE(turbo::tlog::level::to_string_view(turbo::tlog::level::warn) == "warning");
    REQUIRE(turbo::tlog::level::to_string_view(turbo::tlog::level::err) == "error");
    REQUIRE(turbo::tlog::level::to_string_view(turbo::tlog::level::critical) == "critical");
    REQUIRE(turbo::tlog::level::to_string_view(turbo::tlog::level::off) == "off");
}

TEST_CASE("to_short_c_str [convert_to_short_c_str]")
{
    REQUIRE(std::string(turbo::tlog::level::to_short_c_str(turbo::tlog::level::trace)) == "T");
    REQUIRE(std::string(turbo::tlog::level::to_short_c_str(turbo::tlog::level::debug)) == "D");
    REQUIRE(std::string(turbo::tlog::level::to_short_c_str(turbo::tlog::level::info)) == "I");
    REQUIRE(std::string(turbo::tlog::level::to_short_c_str(turbo::tlog::level::warn)) == "W");
    REQUIRE(std::string(turbo::tlog::level::to_short_c_str(turbo::tlog::level::err)) == "E");
    REQUIRE(std::string(turbo::tlog::level::to_short_c_str(turbo::tlog::level::critical)) == "C");
    REQUIRE(std::string(turbo::tlog::level::to_short_c_str(turbo::tlog::level::off)) == "O");
}

TEST_CASE("to_level_enum [convert_to_level_enum]")
{
    REQUIRE(turbo::tlog::level::from_str("trace") == turbo::tlog::level::trace);
    REQUIRE(turbo::tlog::level::from_str("debug") == turbo::tlog::level::debug);
    REQUIRE(turbo::tlog::level::from_str("info") == turbo::tlog::level::info);
    REQUIRE(turbo::tlog::level::from_str("warning") == turbo::tlog::level::warn);
    REQUIRE(turbo::tlog::level::from_str("warn") == turbo::tlog::level::warn);
    REQUIRE(turbo::tlog::level::from_str("error") == turbo::tlog::level::err);
    REQUIRE(turbo::tlog::level::from_str("critical") == turbo::tlog::level::critical);
    REQUIRE(turbo::tlog::level::from_str("off") == turbo::tlog::level::off);
    REQUIRE(turbo::tlog::level::from_str("null") == turbo::tlog::level::off);
}

TEST_CASE("periodic flush [periodic_flush]")
{
    using turbo::tlog::sinks::test_sink_mt;
    auto logger = turbo::tlog::create<test_sink_mt>("periodic_flush");
    auto test_sink = std::static_pointer_cast<test_sink_mt>(logger->sinks()[0]);

    turbo::tlog::flush_every(std::chrono::seconds(1));
    std::this_thread::sleep_for(std::chrono::milliseconds(1250));
    REQUIRE(test_sink->flush_counter() == 1);
    turbo::tlog::flush_every(std::chrono::seconds(0));
    turbo::tlog::drop_all();
}

TEST_CASE("clone-logger [clone]")
{
    using turbo::tlog::sinks::test_sink_mt;
    auto test_sink = std::make_shared<test_sink_mt>();
    auto logger = std::make_shared<turbo::tlog::logger>("orig", test_sink);
    logger->set_pattern("%v");
    auto cloned = logger->clone("clone");

    REQUIRE(cloned->name() == "clone");
    REQUIRE(logger->sinks() == cloned->sinks());
    REQUIRE(logger->level() == cloned->level());
    REQUIRE(logger->flush_level() == cloned->flush_level());
    logger->info("Some message 1");
    cloned->info("Some message 2");

    REQUIRE(test_sink->lines().size() == 2);
    REQUIRE(test_sink->lines()[0] == "Some message 1");
    REQUIRE(test_sink->lines()[1] == "Some message 2");

    turbo::tlog::drop_all();
}

TEST_CASE("clone async [clone]")
{
    using turbo::tlog::sinks::test_sink_st;
    turbo::tlog::init_thread_pool(4, 1);
    auto test_sink = std::make_shared<test_sink_st>();
    auto logger = std::make_shared<turbo::tlog::async_logger>("orig", test_sink, turbo::tlog::thread_pool());
    logger->set_pattern("%v");
    auto cloned = logger->clone("clone");

    REQUIRE(cloned->name() == "clone");
    REQUIRE(logger->sinks() == cloned->sinks());
    REQUIRE(logger->level() == cloned->level());
    REQUIRE(logger->flush_level() == cloned->flush_level());

    logger->info("Some message 1");
    cloned->info("Some message 2");

    turbo::sleep_for(turbo::Duration::milliseconds(100));

    REQUIRE(test_sink->lines().size() == 2);
    REQUIRE(test_sink->lines()[0] == "Some message 1");
    REQUIRE(test_sink->lines()[1] == "Some message 2");

    turbo::tlog::drop_all();
}

TEST_CASE("to_hex [to_hex]")
{
    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);
    turbo::tlog::logger oss_logger("oss", oss_sink);

    std::vector<unsigned char> v{9, 0xa, 0xb, 0xc, 0xff, 0xff};
    oss_logger.info("{}", turbo::tlog::to_hex(v));

    auto output = oss.str();
    REQUIRE(ends_with(output, "0000: 09 0a 0b 0c ff ff" + std::string(turbo::tlog::details::os::default_eol)));
}

TEST_CASE("to_hex_upper [to_hex]")
{
    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);
    turbo::tlog::logger oss_logger("oss", oss_sink);

    std::vector<unsigned char> v{9, 0xa, 0xb, 0xc, 0xff, 0xff};
    oss_logger.info("{:X}", turbo::tlog::to_hex(v));

    auto output = oss.str();
    REQUIRE(ends_with(output, "0000: 09 0A 0B 0C FF FF" + std::string(turbo::tlog::details::os::default_eol)));
}

TEST_CASE("to_hex_no_delimiter [to_hex]")
{
    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);
    turbo::tlog::logger oss_logger("oss", oss_sink);

    std::vector<unsigned char> v{9, 0xa, 0xb, 0xc, 0xff, 0xff};
    oss_logger.info("{:sX}", turbo::tlog::to_hex(v));

    auto output = oss.str();
    REQUIRE(ends_with(output, "0000: 090A0B0CFFFF" + std::string(turbo::tlog::details::os::default_eol)));
}

TEST_CASE("to_hex_show_ascii [to_hex]")
{
    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);
    turbo::tlog::logger oss_logger("oss", oss_sink);

    std::vector<unsigned char> v{9, 0xa, 0xb, 0x41, 0xc, 0x4b, 0xff, 0xff};
    oss_logger.info("{:Xsa}", turbo::tlog::to_hex(v, 8));

    REQUIRE(ends_with(oss.str(), "0000: 090A0B410C4BFFFF  ...A.K.." + std::string(turbo::tlog::details::os::default_eol)));
}

TEST_CASE("to_hex_different_size_per_line [to_hex]")
{
    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);
    turbo::tlog::logger oss_logger("oss", oss_sink);

    std::vector<unsigned char> v{9, 0xa, 0xb, 0x41, 0xc, 0x4b, 0xff, 0xff};

    oss_logger.info("{:Xsa}", turbo::tlog::to_hex(v, 10));
    REQUIRE(ends_with(oss.str(), "0000: 090A0B410C4BFFFF  ...A.K.." + std::string(turbo::tlog::details::os::default_eol)));

    oss_logger.info("{:Xs}", turbo::tlog::to_hex(v, 10));
    REQUIRE(ends_with(oss.str(), "0000: 090A0B410C4BFFFF" + std::string(turbo::tlog::details::os::default_eol)));

    oss_logger.info("{:Xsa}", turbo::tlog::to_hex(v, 6));
    REQUIRE(ends_with(oss.str(), "0000: 090A0B410C4B  ...A.K" + std::string(turbo::tlog::details::os::default_eol) + "0006: FFFF          .." +
                                     std::string(turbo::tlog::details::os::default_eol)));

    oss_logger.info("{:Xs}", turbo::tlog::to_hex(v, 6));
    REQUIRE(ends_with(oss.str(), "0000: 090A0B410C4B" + std::string(turbo::tlog::details::os::default_eol) + "0006: FFFF" +
                                     std::string(turbo::tlog::details::os::default_eol)));
}

TEST_CASE("to_hex_no_ascii [to_hex]")
{
    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);
    turbo::tlog::logger oss_logger("oss", oss_sink);

    std::vector<unsigned char> v{9, 0xa, 0xb, 0x41, 0xc, 0x4b, 0xff, 0xff};
    oss_logger.info("{:Xs}", turbo::tlog::to_hex(v, 8));

    REQUIRE(ends_with(oss.str(), "0000: 090A0B410C4BFFFF" + std::string(turbo::tlog::details::os::default_eol)));

    oss_logger.info("{:Xsna}", turbo::tlog::to_hex(v, 8));

    REQUIRE(ends_with(oss.str(), "090A0B410C4BFFFF" + std::string(turbo::tlog::details::os::default_eol)));
}

TEST_CASE("default logger API [default logger]")
{
    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);

    turbo::tlog::set_default_logger(std::make_shared<turbo::tlog::logger>("oss", oss_sink));
    turbo::tlog::set_pattern("*** %v");

    turbo::tlog::default_logger()->set_level(turbo::tlog::level::trace);
    turbo::tlog::trace("hello trace");
    REQUIRE(oss.str() == "*** hello trace" + std::string(turbo::tlog::details::os::default_eol));

    oss.str("");
    turbo::tlog::debug("hello debug");
    REQUIRE(oss.str() == "*** hello debug" + std::string(turbo::tlog::details::os::default_eol));

    oss.str("");
    turbo::tlog::info("Hello");
    REQUIRE(oss.str() == "*** Hello" + std::string(turbo::tlog::details::os::default_eol));

    oss.str("");
    turbo::tlog::warn("Hello again {}", 2);
    REQUIRE(oss.str() == "*** Hello again 2" + std::string(turbo::tlog::details::os::default_eol));

    oss.str("");
    turbo::tlog::error(123);
    REQUIRE(oss.str() == "*** 123" + std::string(turbo::tlog::details::os::default_eol));

    oss.str("");
    turbo::tlog::critical(std::string("some string"));
    REQUIRE(oss.str() == "*** some string" + std::string(turbo::tlog::details::os::default_eol));

    oss.str("");
    turbo::tlog::set_level(turbo::tlog::level::info);
    turbo::tlog::debug("should not be logged");
    REQUIRE(oss.str().empty());
    turbo::tlog::drop_all();
    turbo::tlog::set_pattern("%v");
}
