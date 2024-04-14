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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "collie/testing/test.h"
#include "includes.h"

#ifdef TLOG_USE_STD_FORMAT
using filename_memory_buf_t = std::basic_string<clog::filename_t::value_type>;
#else
using filename_memory_buf_t = turbo::basic_memory_buffer<clog::filename_t::value_type, 250>;
#endif

#ifdef TLOG_WCHAR_FILENAMES
std::string filename_buf_to_utf8string(const filename_memory_buf_t &w)
{
    clog::memory_buf_t buf;
    clog::details::os::wstr_to_utf8buf(clog::wstring_view_t(w.data(), w.size()), buf);
    return TLOG_BUF_TO_STRING(buf);
}
#else

std::string filename_buf_to_utf8string(const filename_memory_buf_t &w) {
    return TLOG_BUF_TO_STRING(w);
}

#endif

TEST_CASE("daily_logger with dateonly calculator [daily_logger]")
{
    using sink_type = clog::sinks::daily_file_sink<std::mutex, clog::sinks::daily_filename_calculator>;

    prepare_logdir();

    // calculate filename (time based)
    clog::filename_t basename = TLOG_FILENAME_T("test_logs/daily_dateonly");
    std::tm tm = turbo::Time::time_now().to_local_tm();
    filename_memory_buf_t w;
    turbo::format_to(
            std::back_inserter(w), TLOG_FILENAME_T("{}_{:04d}-{:02d}-{:02d}"), basename, tm.tm_year + 1900,
            tm.tm_mon + 1, tm.tm_mday);

    auto logger = clog::create<sink_type>("logger", basename, 0, 0);
    for (int i = 0; i < 10; ++i) {

        logger->info("Test message {}", i);
    }
    logger->flush();

    require_message_count(filename_buf_to_utf8string(w), 10);
}

struct custom_daily_file_name_calculator {
    static clog::filename_t calc_filename(const clog::filename_t &basename, const tm &now_tm) {
        filename_memory_buf_t w;
        turbo::format_to(std::back_inserter(w), TLOG_FILENAME_T("{}{:04d}{:02d}{:02d}"), basename,
                                        now_tm.tm_year + 1900,
                                        now_tm.tm_mon + 1, now_tm.tm_mday);

        return TLOG_BUF_TO_STRING(w);
    }
};

TEST_CASE("daily_logger with custom calculator [daily_logger]")
{
    using sink_type = clog::sinks::daily_file_sink<std::mutex, custom_daily_file_name_calculator>;

    prepare_logdir();

    // calculate filename (time based)
    clog::filename_t basename = TLOG_FILENAME_T("test_logs/daily_dateonly");
    std::tm tm = turbo::Time::time_now().to_local_tm();
    filename_memory_buf_t w;
    turbo::format_to(
            std::back_inserter(w), TLOG_FILENAME_T("{}{:04d}{:02d}{:02d}"), basename, tm.tm_year + 1900, tm.tm_mon + 1,
            tm.tm_mday);

    auto logger = clog::create<sink_type>("logger", basename, 0, 0);
    for (int i = 0; i < 10; ++i) {
        logger->info("Test message {}", i);
    }

    logger->flush();

    require_message_count(filename_buf_to_utf8string(w), 10);
}

/*
 * File name calculations
 */

TEST_CASE("rotating_file_sink::calc_filename1 [rotating_file_sink]]")
{
    auto filename = clog::sinks::rotating_file_sink_st::calc_filename(TLOG_FILENAME_T("rotated.txt"), 3);
    REQUIRE_EQ(filename , TLOG_FILENAME_T("rotated.3.txt"));
}

TEST_CASE("rotating_file_sink::calc_filename2 [rotating_file_sink]]")
{
    auto filename = clog::sinks::rotating_file_sink_st::calc_filename(TLOG_FILENAME_T("rotated"), 3);
    REQUIRE_EQ(filename , TLOG_FILENAME_T("rotated.3"));
}

TEST_CASE("rotating_file_sink::calc_filename3 [rotating_file_sink]]")
{
    auto filename = clog::sinks::rotating_file_sink_st::calc_filename(TLOG_FILENAME_T("rotated.txt"), 0);
    REQUIRE_EQ(filename , TLOG_FILENAME_T("rotated.txt"));
}

// regex supported only from gcc 4.9 and above
#if defined(_MSC_VER) || !(__GNUC__ <= 4 && __GNUC_MINOR__ < 9)

#    include <regex>

TEST_CASE("daily_file_sink::daily_filename_calculator [daily_file_sink]]")
{
    // daily_YYYY-MM-DD_hh-mm.txt
    auto filename =
            clog::sinks::daily_filename_calculator::calc_filename(TLOG_FILENAME_T("daily.txt"),
                                                                         turbo::Time::time_now().to_local_tm());
    // date regex based on https://www.regular-expressions.info/dates.html
    std::basic_regex<clog::filename_t::value_type> re(
            TLOG_FILENAME_T(R"(^daily_(19|20)\d\d-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])\.txt$)"));
    std::match_results<clog::filename_t::const_iterator> match;
    REQUIRE(std::regex_match(filename, match, re));
}

#endif
/*
TEST_CASE("daily_file_sink::daily_filename_format_calculator [daily_file_sink]]")
{
    std::tm tm = clog::details::os::localtime();
    // example-YYYY-MM-DD.log
    auto filename = clog::sinks::daily_filename_format_calculator::calc_filename(TLOG_FILENAME_T("example-%Y-%m-%d.log"), tm);

    REQUIRE_EQ(filename,
            turbo::format(TLOG_FILENAME_T("example-{:04d}-{:02d}-{:02d}.log"), tm.tm_year + 1900,
                                         tm.tm_mon + 1, tm.tm_mday));
}*/

/* Test removal of old files */
static clog::details::log_msg create_msg(turbo::Duration offset) {
    clog::details::log_msg msg{"test", clog::level::info, "Hello Message"};
    msg.time = turbo::Time::time_now() + offset;
    return msg;
}

static void test_rotate(int days_to_run, uint16_t max_days, uint16_t expected_n_files) {
    using clog::details::log_msg;
    using clog::sinks::daily_file_sink_st;

    prepare_logdir();

    clog::filename_t basename = TLOG_FILENAME_T("test_logs/daily_rotate.txt");
    daily_file_sink_st sink{basename, 2, 30, true, max_days};

    // simulate messages with 24 intervals

    for (int i = 0; i < days_to_run; i++) {
        auto offset = turbo::Duration::seconds(24 * 3600 * i);
        sink.log(create_msg(offset));
    }

    REQUIRE_EQ(count_files("test_logs") , static_cast<size_t>(expected_n_files));
}

TEST_CASE("daily_logger rotate [daily_file_sink]")
{
    int days_to_run = 1;
    test_rotate(days_to_run, 0, 1);
    test_rotate(days_to_run, 1, 1);
    test_rotate(days_to_run, 3, 1);
    test_rotate(days_to_run, 10, 1);

    days_to_run = 10;
    test_rotate(days_to_run, 0, 10);
    test_rotate(days_to_run, 1, 1);
    test_rotate(days_to_run, 3, 3);
    test_rotate(days_to_run, 9, 9);
    test_rotate(days_to_run, 10, 10);
    test_rotate(days_to_run, 11, 10);
    test_rotate(days_to_run, 20, 10);
}
