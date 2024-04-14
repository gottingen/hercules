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

using turbo::tlog::memory_buf_t;
using turbo::tlog::details::fmt_helper::to_string_view;

// log to str and return it
template<typename... Args>
static std::string log_to_str(const std::string &msg, const Args &... args) {
    std::ostringstream oss;
    auto oss_sink = std::make_shared<turbo::tlog::sinks::ostream_sink_mt>(oss);
    turbo::tlog::logger oss_logger("pattern_tester", oss_sink);
    oss_logger.set_level(turbo::tlog::level::info);

    oss_logger.set_formatter(std::unique_ptr<turbo::tlog::formatter>(new turbo::tlog::pattern_formatter(args...)));

    oss_logger.info(msg);
    return oss.str();
}

TEST_CASE("custom eol [pattern_formatter]")
{
    std::string msg = "Hello custom eol test";
    std::string eol = ";)";
    REQUIRE(log_to_str(msg, "%v", turbo::tlog::pattern_time_type::local, ";)") == msg + eol);
}

TEST_CASE("empty format [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "", turbo::tlog::pattern_time_type::local, "").empty());
}

TEST_CASE("empty format2 [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "", turbo::tlog::pattern_time_type::local, "\n") == "\n");
}

TEST_CASE("level [pattern_formatter]")
{
    REQUIRE_EQ(log_to_str("Some message", "[%l] %v", turbo::tlog::pattern_time_type::local, "\n"),
               "[info] Some message\n");
}

TEST_CASE("short level [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%L] %v", turbo::tlog::pattern_time_type::local, "\n") == "[I] Some message\n");
}

TEST_CASE("name [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pattern_tester] Some message\n");
}

TEST_CASE("date MM/DD/YY  [pattern_formatter]")
{
    auto now_tm = turbo::Time::time_now().to_local_tm();
    std::stringstream oss;
    oss << std::setfill('0') << std::setw(2) << now_tm.tm_mon + 1 << "/" << std::setw(2) << now_tm.tm_mday << "/"
        << std::setw(2)
        << (now_tm.tm_year + 1900) % 1000 << " Some message\n";
    REQUIRE(log_to_str("Some message", "%D %v", turbo::tlog::pattern_time_type::local, "\n") == oss.str());
}

TEST_CASE("color range test1 [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>("%^%v%$", turbo::tlog::pattern_time_type::local,
                                                                      "\n");

    memory_buf_t buf;
    turbo::format_to(std::back_inserter(buf), "Hello");
    memory_buf_t formatted;
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info,
                                      std::string_view(buf.data(), buf.size()));
    formatter->format(msg, formatted);
    REQUIRE_EQ(msg.color_range_start , 0);
    REQUIRE_EQ(msg.color_range_end , 5);
    REQUIRE_EQ(log_to_str("hello", "%^%v%$", turbo::tlog::pattern_time_type::local, "\n") , "hello\n");
}

TEST_CASE("color range test2 [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>("%^%$", turbo::tlog::pattern_time_type::local,
                                                                      "\n");
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "");
    memory_buf_t formatted;
    formatter->format(msg, formatted);
    REQUIRE_EQ(msg.color_range_start , 0);
    REQUIRE_EQ(msg.color_range_end , 0);
    REQUIRE_EQ(log_to_str("", "%^%$", turbo::tlog::pattern_time_type::local, "\n") , "\n");
}

TEST_CASE("color range test3 [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>("%^***%$");
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "ignored");
    memory_buf_t formatted;
    formatter->format(msg, formatted);
    REQUIRE_EQ(msg.color_range_start , 0);
    REQUIRE_EQ(msg.color_range_end , 3);
}

TEST_CASE("color range test4 [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>("XX%^YYY%$",
                                                                      turbo::tlog::pattern_time_type::local, "\n");
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "ignored");

    memory_buf_t formatted;
    formatter->format(msg, formatted);
    REQUIRE_EQ(msg.color_range_start , 2);
    REQUIRE_EQ(msg.color_range_end , 5);
    REQUIRE_EQ(log_to_str("ignored", "XX%^YYY%$", turbo::tlog::pattern_time_type::local, "\n") , "XXYYY\n");
}

TEST_CASE("color range test5 [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>("**%^");
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "ignored");
    memory_buf_t formatted;
    formatter->format(msg, formatted);
    REQUIRE_EQ(msg.color_range_start , 2);
    REQUIRE_EQ(msg.color_range_end , 0);
}

TEST_CASE("color range test6 [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>("**%$");
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "ignored");
    memory_buf_t formatted;
    formatter->format(msg, formatted);
    REQUIRE_EQ(msg.color_range_start, 0);
    REQUIRE_EQ(msg.color_range_end, 2);
}

//
// Test padding
//

TEST_CASE("level_left_padded [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%8l] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[    info] Some message\n");
    REQUIRE(log_to_str("Some message", "[%8!l] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[    info] Some message\n");
}

TEST_CASE("level_right_padded [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%-8l] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[info    ] Some message\n");
    REQUIRE(log_to_str("Some message", "[%-8!l] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[info    ] Some message\n");
}

TEST_CASE("level_center_padded [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%=8l] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[  info  ] Some message\n");
    REQUIRE(log_to_str("Some message", "[%=8!l] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[  info  ] Some message\n");
}

TEST_CASE("short level_left_padded [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%3L] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[  I] Some message\n");
    REQUIRE(log_to_str("Some message", "[%3!L] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[  I] Some message\n");
}

TEST_CASE("short level_right_padded [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%-3L] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[I  ] Some message\n");
    REQUIRE(log_to_str("Some message", "[%-3!L] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[I  ] Some message\n");
}

TEST_CASE("short level_center_padded [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%=3L] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[ I ] Some message\n");
    REQUIRE(log_to_str("Some message", "[%=3!L] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[ I ] Some message\n");
}

TEST_CASE("left_padded_short [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%3n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pattern_tester] Some message\n");
    REQUIRE(log_to_str("Some message", "[%3!n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pat] Some message\n");
}

TEST_CASE("right_padded_short [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%-3n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pattern_tester] Some message\n");
    REQUIRE(log_to_str("Some message", "[%-3!n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pat] Some message\n");
}

TEST_CASE("center_padded_short [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%=3n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pattern_tester] Some message\n");
    REQUIRE(log_to_str("Some message", "[%=3!n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pat] Some message\n");
}

TEST_CASE("left_padded_huge [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%-300n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pattern_tester                                                  ] Some message\n");

    REQUIRE(log_to_str("Some message", "[%-300!n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pattern_tester                                                  ] Some message\n");
}

TEST_CASE("left_padded_max [pattern_formatter]")
{
    REQUIRE(log_to_str("Some message", "[%-64n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pattern_tester                                                  ] Some message\n");

    REQUIRE(log_to_str("Some message", "[%-64!n] %v", turbo::tlog::pattern_time_type::local, "\n") ==
            "[pattern_tester                                                  ] Some message\n");
}

// Test padding + truncate flag

TEST_CASE("paddinng_truncate [pattern_formatter]")
{
    REQUIRE(log_to_str("123456", "%6!v", turbo::tlog::pattern_time_type::local, "\n") == "123456\n");
    REQUIRE(log_to_str("123456", "%5!v", turbo::tlog::pattern_time_type::local, "\n") == "12345\n");
    REQUIRE(log_to_str("123456", "%7!v", turbo::tlog::pattern_time_type::local, "\n") == " 123456\n");

    REQUIRE(log_to_str("123456", "%-6!v", turbo::tlog::pattern_time_type::local, "\n") == "123456\n");
    REQUIRE(log_to_str("123456", "%-5!v", turbo::tlog::pattern_time_type::local, "\n") == "12345\n");
    REQUIRE(log_to_str("123456", "%-7!v", turbo::tlog::pattern_time_type::local, "\n") == "123456 \n");

    REQUIRE(log_to_str("123456", "%=6!v", turbo::tlog::pattern_time_type::local, "\n") == "123456\n");
    REQUIRE(log_to_str("123456", "%=5!v", turbo::tlog::pattern_time_type::local, "\n") == "12345\n");
    REQUIRE(log_to_str("123456", "%=7!v", turbo::tlog::pattern_time_type::local, "\n") == "123456 \n");

    REQUIRE(log_to_str("123456", "%0!v", turbo::tlog::pattern_time_type::local, "\n") == "\n");
}

TEST_CASE("padding_truncate_funcname [pattern_formatter]")
{
    turbo::tlog::sinks::test_sink_st test_sink;

    const char *pattern = "%v [%5!!]";
    auto formatter = std::unique_ptr<turbo::tlog::formatter>(new turbo::tlog::pattern_formatter(pattern));
    test_sink.set_formatter(std::move(formatter));

    turbo::tlog::details::log_msg msg1{turbo::tlog::source_loc{"ignored", 1, "func"}, "test_logger",
                                       turbo::tlog::level::info, "message"};
    test_sink.log(msg1);
    REQUIRE_EQ(test_sink.lines()[0], "message [ func]");

    turbo::tlog::details::log_msg msg2{turbo::tlog::source_loc{"ignored", 1, "function"}, "test_logger",
                                       turbo::tlog::level::info, "message"};
    test_sink.log(msg2);
    REQUIRE_EQ(test_sink.lines()[1], "message [funct]");
}

TEST_CASE("padding_funcname [pattern_formatter]")
{
    turbo::tlog::sinks::test_sink_st test_sink;

    const char *pattern = "%v [%10!]";
    auto formatter = std::unique_ptr<turbo::tlog::formatter>(new turbo::tlog::pattern_formatter(pattern));
    test_sink.set_formatter(std::move(formatter));

    turbo::tlog::details::log_msg msg1{turbo::tlog::source_loc{"ignored", 1, "func"}, "test_logger",
                                       turbo::tlog::level::info, "message"};
    test_sink.log(msg1);
    REQUIRE_EQ(test_sink.lines()[0], "message [      func]");

    turbo::tlog::details::log_msg msg2{turbo::tlog::source_loc{"ignored", 1, "func567890123"}, "test_logger",
                                       turbo::tlog::level::info, "message"};
    test_sink.log(msg2);
    REQUIRE_EQ(test_sink.lines()[1], "message [func567890123]");
}

TEST_CASE("clone-default-formatter [pattern_formatter]")
{
    auto formatter_1 = std::make_shared<turbo::tlog::pattern_formatter>();
    auto formatter_2 = formatter_1->clone();
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "some message");

    memory_buf_t formatted_1;
    memory_buf_t formatted_2;
    formatter_1->format(msg, formatted_1);
    formatter_2->format(msg, formatted_2);

    REQUIRE_EQ(to_string_view(formatted_1), to_string_view(formatted_2));
}

TEST_CASE("clone-default-formatter2 [pattern_formatter]")
{
    auto formatter_1 = std::make_shared<turbo::tlog::pattern_formatter>("%+");
    auto formatter_2 = formatter_1->clone();
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "some message");

    memory_buf_t formatted_1;
    memory_buf_t formatted_2;
    formatter_1->format(msg, formatted_1);
    formatter_2->format(msg, formatted_2);

    REQUIRE_EQ(to_string_view(formatted_1), to_string_view(formatted_2));
}

TEST_CASE("clone-formatter [pattern_formatter]")
{
    auto formatter_1 = std::make_shared<turbo::tlog::pattern_formatter>("%D %X [%] [%n] %v");
    auto formatter_2 = formatter_1->clone();
    std::string logger_name = "test";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "some message");

    memory_buf_t formatted_1;
    memory_buf_t formatted_2;
    formatter_1->format(msg, formatted_1);
    formatter_2->format(msg, formatted_2);

    REQUIRE_EQ(to_string_view(formatted_1), to_string_view(formatted_2));
}

TEST_CASE("clone-formatter-2 [pattern_formatter]")
{
    using turbo::tlog::pattern_time_type;
    auto formatter_1 = std::make_shared<turbo::tlog::pattern_formatter>("%D %X [%] [%n] %v", pattern_time_type::utc,
                                                                        "xxxxxx\n");
    auto formatter_2 = formatter_1->clone();
    std::string logger_name = "test2";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "some message");

    memory_buf_t formatted_1;
    memory_buf_t formatted_2;
    formatter_1->format(msg, formatted_1);
    formatter_2->format(msg, formatted_2);

    REQUIRE_EQ(to_string_view(formatted_1), to_string_view(formatted_2));
}

class custom_test_flag : public turbo::tlog::custom_flag_formatter {
public:
    explicit custom_test_flag(std::string txt)
            : some_txt{std::move(txt)} {}

    void format(const turbo::tlog::details::log_msg &, const turbo::CivilInfo &tm, turbo::tlog::memory_buf_t &dest) override {
        if (some_txt == "throw_me") {
            throw turbo::tlog::tlog_ex("custom_flag_exception_test");
        } else if (some_txt == "time") {
            auto formatted = turbo::format("{:d}:{:02d}{:s}", tm.hour() % 12, tm.minute(),
                                                          tm.hour() / 12 ? "PM" : "AM");
            dest.append(formatted.data(), formatted.data() + formatted.size());
            return;
        }
        some_txt = std::string(padinfo_.width_, ' ') + some_txt;
        dest.append(some_txt.data(), some_txt.data() + some_txt.size());
    }

    turbo::tlog::details::padding_info get_padding_info() {
        return padinfo_;
    }

    std::string some_txt;

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return turbo::tlog::details::make_unique<custom_test_flag>(some_txt);
    }
};
// test clone with custom flag formatters
TEST_CASE("clone-custom_formatter [pattern_formatter]")
{
    auto formatter_1 = std::make_shared<turbo::tlog::pattern_formatter>();
    formatter_1->add_flag<custom_test_flag>('t', "custom_output").set_pattern("[%n] [%t] %v");
    auto formatter_2 = formatter_1->clone();
    std::string logger_name = "logger-name";
    turbo::tlog::details::log_msg msg(logger_name, turbo::tlog::level::info, "some message");

    memory_buf_t formatted_1;
    memory_buf_t formatted_2;
    formatter_1->format(msg, formatted_1);
    formatter_2->format(msg, formatted_2);

    auto expected = turbo::format("[logger-name] [custom_output] some message{}",
                                                 turbo::tlog::details::os::default_eol);

    REQUIRE_EQ(to_string_view(formatted_1), expected);
    REQUIRE_EQ(to_string_view(formatted_2), expected);
}

//
// Test source location formatting
//

#ifdef _WIN32
static const char *const test_path = "\\a\\b\\c/myfile.cpp";
#else
static const char *const test_path = "/a/b//myfile.cpp";
#endif

TEST_CASE("short filename formatter-1 [pattern_formatter]")
{
    turbo::tlog::pattern_formatter formatter("%s", turbo::tlog::pattern_time_type::local, "");
    memory_buf_t formatted;
    std::string logger_name = "logger-name";
    turbo::tlog::source_loc source_loc{test_path, 123, "some_func()"};
    turbo::tlog::details::log_msg msg(source_loc, "logger-name", turbo::tlog::level::info, "Hello");
    formatter.format(msg, formatted);

    REQUIRE_EQ(to_string_view(formatted), "myfile.cpp");
}

TEST_CASE("short filename formatter-2 [pattern_formatter]")
{
    turbo::tlog::pattern_formatter formatter("%s:%#", turbo::tlog::pattern_time_type::local, "");
    memory_buf_t formatted;
    std::string logger_name = "logger-name";
    turbo::tlog::source_loc source_loc{"myfile.cpp", 123, "some_func()"};
    turbo::tlog::details::log_msg msg(source_loc, "logger-name", turbo::tlog::level::info, "Hello");
    formatter.format(msg, formatted);

    REQUIRE_EQ(to_string_view(formatted), "myfile.cpp:123");
}

TEST_CASE("short filename formatter-3 [pattern_formatter]")
{
    turbo::tlog::pattern_formatter formatter("%s %v", turbo::tlog::pattern_time_type::local, "");
    memory_buf_t formatted;
    std::string logger_name = "logger-name";
    turbo::tlog::source_loc source_loc{"", 123, "some_func()"};
    turbo::tlog::details::log_msg msg(source_loc, "logger-name", turbo::tlog::level::info, "Hello");
    formatter.format(msg, formatted);

    REQUIRE_EQ(to_string_view(formatted), " Hello");
}

TEST_CASE("full filename formatter [pattern_formatter]")
{
    turbo::tlog::pattern_formatter formatter("%g", turbo::tlog::pattern_time_type::local, "");
    memory_buf_t formatted;
    std::string logger_name = "logger-name";
    turbo::tlog::source_loc source_loc{test_path, 123, "some_func()"};
    turbo::tlog::details::log_msg msg(source_loc, "logger-name", turbo::tlog::level::info, "Hello");
    formatter.format(msg, formatted);

    REQUIRE_EQ(to_string_view(formatted), test_path);
}

TEST_CASE("custom flags [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>();
    formatter->add_flag<custom_test_flag>('t', "custom1").add_flag<custom_test_flag>('u', "custom2").set_pattern(
            "[%n] [%t] [%u] %v");

    memory_buf_t formatted;

    turbo::tlog::details::log_msg msg(turbo::tlog::source_loc{}, "logger-name", turbo::tlog::level::info,
                                      "some message");
    formatter->format(msg, formatted);
    auto expected = turbo::format("[logger-name] [custom1] [custom2] some message{}",
                                                 turbo::tlog::details::os::default_eol);

    REQUIRE_EQ(to_string_view(formatted), expected);
}

TEST_CASE("custom flags-padding [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>();
    formatter->add_flag<custom_test_flag>('t', "custom1").add_flag<custom_test_flag>('u', "custom2").set_pattern(
            "[%n] [%t] [%5u] %v");

    memory_buf_t formatted;

    turbo::tlog::details::log_msg msg(turbo::tlog::source_loc{}, "logger-name", turbo::tlog::level::info,
                                      "some message");
    formatter->format(msg, formatted);
    auto expected = turbo::format("[logger-name] [custom1] [     custom2] some message{}",
                                                 turbo::tlog::details::os::default_eol);

    REQUIRE_EQ(to_string_view(formatted), expected);
}

TEST_CASE("custom flags-exception [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>();
    formatter->add_flag<custom_test_flag>('t', "throw_me").add_flag<custom_test_flag>('u', "custom2").set_pattern(
            "[%n] [%t] [%u] %v");

    memory_buf_t formatted;
    turbo::tlog::details::log_msg msg(turbo::tlog::source_loc{}, "logger-name", turbo::tlog::level::info,
                                      "some message");
    CHECK_THROWS_AS(formatter->format(msg, formatted), turbo::tlog::tlog_ex);
}

TEST_CASE("override need_localtime [pattern_formatter]")
{
    auto formatter = std::make_shared<turbo::tlog::pattern_formatter>(turbo::tlog::pattern_time_type::local, "\n");
    formatter->add_flag<custom_test_flag>('t', "time").set_pattern("%t> %v");

    {
        memory_buf_t formatted;
        turbo::tlog::details::log_msg msg(turbo::tlog::source_loc{}, "logger-name", turbo::tlog::level::info,
                                          "some message");
        formatter->format(msg, formatted);
        REQUIRE_EQ(to_string_view(formatted), "0:00AM> some message\n");
    }

    {
        formatter->need_localtime();

        auto now_tm = turbo::time_now().to_local_tm();
        std::stringstream oss;
        oss << (now_tm.tm_hour % 12) << ":" << std::setfill('0') << std::setw(2) << now_tm.tm_min
            << (now_tm.tm_hour / 12 ? "PM" : "AM")
            << "> some message\n";

        memory_buf_t formatted;
        turbo::tlog::details::log_msg msg(turbo::tlog::source_loc{}, "logger-name", turbo::tlog::level::info,
                                          "some message");
        formatter->format(msg, formatted);
        REQUIRE_EQ(to_string_view(formatted), oss.str());
    }
}
