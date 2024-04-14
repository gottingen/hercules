//
// Copyright(c) 2015 Gabi Melman.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

// spdlog usage example

#include <cstdio>
#include <chrono>
#include "turbo/format/format.h"
#include "turbo/log/tlog.h"
#include "turbo/log/cfg/env.h"  // support for loading levels from the environment variable
#include "turbo/log/sinks/basic_file_sink.h"
#include "turbo/log/sinks/stdout_color_sinks.h"
#include "turbo/log/async.h"
#include "turbo/log/sinks/rotating_file_sink.h"
#include "turbo/log/pattern_formatter.h"
#include <thread>
#include "turbo/log/fmt/bin_to_hex.h"
#include "turbo/log/sinks/daily_file_sink.h"
#include "turbo/log/cfg/env.h"
#include "turbo/log/sinks/udp_sink.h"
#include "turbo/log/logging.h"
#include "turbo/times/stop_watcher.h"
#include "turbo/files/filesystem.h"


void load_levels_example();

void stdout_logger_example();

void basic_example();

void rotating_example();

void daily_example();

void async_example();

void binary_example();

void vector_example();

void stopwatch_example();

void trace_example();

void multi_sink_example();

void user_defined_example();

void err_handler_example();

void syslog_example();

void udp_example();

void custom_flags_example();

void file_events_example();

void replace_default_logger_example();


int main(int, char *[]) {
    // Log levels can be loaded from argv/env using "TLOG_LEVEL"
    if(turbo::filesystem::exists("logs")) {
        turbo::filesystem::remove_all("logs");
    }
    load_levels_example();

    turbo::tlog::warn("Easy padding in numbers like {:08d}", 12);
    turbo::tlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
    turbo::tlog::info("Support for floats {:03.2f}", 1.23456);
    turbo::tlog::info("Positional args are {1} {0}..", "too", "supported");
    turbo::tlog::info("{:>8} aligned, {:<8} aligned", "right", "left");

    // Runtime log levels
    turbo::tlog::set_level(turbo::tlog::level::info); // Set global log level to info
    turbo::tlog::debug("This message should not be displayed!");
    turbo::tlog::set_level(turbo::tlog::level::trace); // Set specific logger's log level
    turbo::tlog::debug("This message should be displayed..");

    // Customize msg format for all loggers
    turbo::tlog::set_pattern("[%H:%M:%S %z] [%^%L%$] [thread %t] %v");
    turbo::tlog::info("This an info message with custom format");
    turbo::tlog::set_pattern("%+"); // back to default format
    turbo::tlog::set_level(turbo::tlog::level::info);

    // Backtrace support
    // Loggers can store in a ring buffer all messages (including debug/trace) for later inspection.
    // When needed, call dump_backtrace() to see what happened:
    turbo::tlog::enable_backtrace(10); // create ring buffer with capacity of 10  messages
    for (int i = 0; i < 100; i++) {
        turbo::tlog::debug("Backtrace message {}", i); // not logged..
    }
    // e.g. if some error happened:
    turbo::tlog::dump_backtrace(); // log them now!

    try {
        stdout_logger_example();
        basic_example();
        rotating_example();
        daily_example();
        async_example();
        binary_example();
        vector_example();
        multi_sink_example();
        user_defined_example();
        err_handler_example();
        trace_example();
        stopwatch_example();
        udp_example();
        custom_flags_example();
        file_events_example();
        replace_default_logger_example();

        // Flush all *registered* loggers using a worker thread every 3 seconds.
        // note: registered loggers *must* be thread safe for this to work correctly!
        turbo::tlog::flush_every(std::chrono::seconds(3));

        // Apply some function on all registered loggers
        turbo::tlog::apply_all([&](std::shared_ptr<turbo::tlog::logger> l) { l->info("End of example."); });

        // Release all spdlog resources, and drop all loggers in the registry.
        // This is optional (only mandatory if using windows + async log).
        turbo::tlog::shutdown();
    }

        // Exceptions will only be thrown upon failed logger or sink construction (not during logging).
    catch (const turbo::tlog::tlog_ex &ex) {
        std::printf("Log initialization failed: %s\n", ex.what());
        return 1;
    }
}


void stdout_logger_example() {
    // Create color multi threaded logger.
    auto console = turbo::tlog::stdout_color_mt("console");
    // or for stderr:
    // auto console = turbo::tlog::stderr_color_mt("error-logger");
}


void basic_example() {
    // Create basic file logger (not rotated).
    auto my_logger = turbo::tlog::basic_logger_mt("file_logger", "logs/basic-log.txt", true);
}


void rotating_example() {
    // Create a file rotating logger with 5mb size max and 3 rotated files.
    auto rotating_logger = turbo::tlog::rotating_logger_mt("some_logger_name", "logs/rotating.txt", 1048576 * 5, 3);
}


void daily_example() {
    // Create a daily logger - a new file is created every day on 2:30am.
    auto daily_logger = turbo::tlog::daily_logger_mt("daily_logger", "logs/daily.txt", 2, 30);
}


void load_levels_example() {
    // Set the log level to "info" and mylogger to "trace":
    // TLOG_LEVEL=info,mylogger=trace && ./example
    turbo::tlog::cfg::load_env_levels();
    // or from command line:
    // ./example TLOG_LEVEL=info,mylogger=trace
    // #include "turbo/log/cfg/argv.h" // for loading levels from argv
    // turbo::tlog::cfg::load_argv_levels(args, argv);
}


void async_example() {
    // Default thread pool settings can be modified *before* creating the async logger:
    // turbo::tlog::init_thread_pool(32768, 1); // queue with max 32k items 1 backing thread.
    auto async_file = turbo::tlog::basic_logger_mt<turbo::tlog::async_factory>("async_file_logger",
                                                                               "logs/async_log.txt");
    // alternatively:
    // auto async_file = turbo::tlog::create_async<turbo::tlog::sinks::basic_file_sink_mt>("async_file_logger", "logs/async_log.txt");

    for (int i = 1; i < 101; ++i) {
        async_file->info("Async message #{}", i);
    }
}

// Log binary data as hex.
// Many types of std::container<char> types can be used.
// Iterator ranges are supported too.
// format flags:
// {:X} - print in uppercase.
// {:s} - don't separate each byte with space.
// {:p} - don't print the position on each line start.
// {:n} - don't split the output to lines.


void binary_example() {
    std::vector<char> buf(80);
    for (int i = 0; i < 80; i++) {
        buf.push_back(static_cast<char>(i & 0xff));
    }
    turbo::tlog::info("Binary example: {}", turbo::tlog::to_hex(buf));
    turbo::tlog::info("Another binary example:{:n}", turbo::tlog::to_hex(std::begin(buf), std::begin(buf) + 10));
    // more examples:
    // logger->info("uppercase: {:X}", turbo::tlog::to_hex(buf));
    // logger->info("uppercase, no delimiters: {:Xs}", turbo::tlog::to_hex(buf));
    // logger->info("uppercase, no delimiters, no position info: {:Xsp}", turbo::tlog::to_hex(buf));
    // logger->info("hexdump style: {:a}", turbo::tlog::to_hex(buf));
    // logger->info("hexdump style, 20 chars per line {:a}", turbo::tlog::to_hex(buf, 20));
}

// Log a vector of numbers

void vector_example() {
    std::vector<int> vec = {1, 2, 3};
    turbo::tlog::info("Vector example: {}", vec);
}

// Compile time log levels.
// define TLOG_ACTIVE_LEVEL to required level (e.g. TLOG_LEVEL_TRACE)
void trace_example() {
    // trace from default logger
    TLOG_TRACE("Some trace message.. {} ,{}", 1, 3.23);
    // debug from default logger
    TLOG_DEBUG("Some debug message.. {} ,{}", 1, 3.23);
    TLOG_INFO("Some info message.. {} ,{}", 1, 3.23);
    TLOG_WARN("Some info message.. {} ,{}", 1, 3.23);
    TLOG_TRACE_IF(true, "Some info message.. {} ,{}", 1, 3.23);
    TLOG_DEBUG_IF(true, "Some info message.. {} ,{}", 1, 3.23);
    TLOG_INFO_IF(true, "Some info message.. {} ,{}", 1, 3.23);
    TLOG_WARN_IF(true, "Some info message.. {} ,{}", 1, 3.23);
    TLOG_ERROR_IF(true, "Some info message.. {} ,{}", 1, 3.23);
    TLOG_CRITICAL_IF(true, "Some info message.. {} ,{}", 1, 3.23);
    TLOG_INFO_IF(false, "this should not display, Some info message.. {} ,{}", 1, 3.23);
    TLOG_WARN_IF(false, "this should not display, Some info message.. {} ,{}", 1, 3.23);
    TLOG_ERROR_IF(false, "this should not display, Some info message.. {} ,{}", 1, 3.23);

    for (size_t i = 0; i < 10; i++) {
        TLOG_TRACE_EVERY_N(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_DEBUG_EVERY_N(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_INFO_EVERY_N(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_WARN_EVERY_N(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_ERROR_EVERY_N(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_CRITICAL_EVERY_N(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
    }

    for (size_t i = 0; i < 10; i++) {
        TLOG_TRACE_FIRST_N(1, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_DEBUG_FIRST_N(1, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_INFO_FIRST_N(1, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_WARN_FIRST_N(1, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_ERROR_FIRST_N(1, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_CRITICAL_FIRST_N(1, "this should display once, Some info message.. {} ,{}", 1, 3.23);
    }

    for (size_t i = 0; i < 10; i++) {
        TLOG_TRACE_ONCE("this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_DEBUG_ONCE("this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_INFO_ONCE("this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_WARN_ONCE("this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_ERROR_ONCE("this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_CRITICAL_ONCE("this should display once, Some info message.. {} ,{}", 1, 3.23);
    }

    for (size_t i = 0; i < 10; i++) {
        // 10 sec is enough not round-trip
        TLOG_TRACE_EVERY_N_SEC(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_DEBUG_EVERY_N_SEC(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_INFO_EVERY_N_SEC(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_WARN_EVERY_N_SEC(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_ERROR_EVERY_N_SEC(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
        TLOG_CRITICAL_EVERY_N_SEC(10, "this should display once, Some info message.. {} ,{}", 1, 3.23);
    }

    // trace from logger object
    auto logger = turbo::tlog::get("file_logger");
    TLOG_LOGGER_TRACE(logger, "another trace message");
    TLOG_CHECK(true);
    TLOG_CHECK(true, "abc");
    TLOG_CHECK(false, "abc{}", 42);
    TLOG_CHECK_EQ(12, 12);
    TLOG_CHECK_EQ(12, 11, "aaa {}", "some reason");
    TLOG_CHECK_NE(12, 11, "aaa {}", "some reason");
    TLOG_CHECK_GT(12, 11, "aaa {}", "some reason");
    TLOG_CHECK_GE(12, 11, "aaa {}", "some reason");
    TLOG_CHECK_LT(10, 11, "aaa {}", "some reason");
    TLOG_CHECK_LE(10, 11, "aaa {}", "some reason");
    TDLOG_CHECK_LE(12, 11, "this should not display {}", "some reason");
    int a = 10;
    size_t bb = 12;
    TLOG_CHECK_EQ(a, bb);
    TLOG_CHECK(1 == 2);
}

// stopwatch example


void stopwatch_example() {
    turbo::StopWatcher sw;
    std::this_thread::sleep_for(std::chrono::milliseconds(123));
    turbo::tlog::info("Stopwatch: {} seconds", sw);
}


void udp_example() {
    turbo::tlog::sinks::udp_sink_config cfg("127.0.0.1", 11091);
    auto my_logger = turbo::tlog::udp_logger_mt("udplog", cfg);
    my_logger->set_level(turbo::tlog::level::debug);
    my_logger->info("hello world");
}

// A logger with multiple sinks (stdout and file) - each with a different format and log level.
void multi_sink_example() {
    auto console_sink = std::make_shared<turbo::tlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(turbo::tlog::level::warn);
    console_sink->set_pattern("[multi_sink_example] [%^%l%$] %v");

    auto file_sink = std::make_shared<turbo::tlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
    file_sink->set_level(turbo::tlog::level::trace);

    turbo::tlog::logger logger("multi_sink", {console_sink, file_sink});
    logger.set_level(turbo::tlog::level::debug);
    logger.warn("this should appear in both console and file");
    logger.info("this message should not appear in the console, only in the file");
}

// User defined types logging
struct my_type {
    int i = 0;

    explicit my_type(int i)
            : i(i) {};
};

template<>
struct turbo::formatter<my_type> : turbo::formatter<std::string> {
    auto format(my_type my, format_context &ctx) -> decltype(ctx.out()) {
        return format_to(ctx.out(), "[my_type i={}]", my.i);
    }
};

void user_defined_example() {
    turbo::tlog::info("user defined type: {}", my_type(14));
}

// Custom error handler. Will be triggered on log failure.
void err_handler_example() {
    // can be set globally or per logger(logger->set_error_handler(..))
    turbo::tlog::set_error_handler(
            [](const std::string &msg) { printf("*** Custom log error handler: %s ***\n", msg.c_str()); });
}

// syslog example (linux/osx/freebsd)
#ifndef _WIN32

#    include "turbo/log/sinks/syslog_sink.h"

void syslog_example() {
    std::string ident = "tlog-example";
    auto syslog_logger = turbo::tlog::syslog_logger_mt("syslog", ident, LOG_PID);
    syslog_logger->warn("This is warning that will end up in syslog.");
}

#endif

// Android example.
#if defined(__ANDROID__)
#    include "turbo/log/sinks/android_sink.h"
void android_example()
{
    std::string tag = "tlog-android";
    auto android_logger = turbo::tlog::android_logger_mt("android", tag);
    android_logger->critical("Use \"adb shell logcat\" to view this message.");
}
#endif

// Log patterns can contain custom flags.
// this will add custom flag '%*' which will be bound to a <my_formatter_flag> instance

class my_formatter_flag : public turbo::tlog::custom_flag_formatter {
public:
    void format(const turbo::tlog::details::log_msg &, const turbo::CivilInfo &, turbo::tlog::memory_buf_t &dest) override {
        std::string some_txt = "custom-flag";
        dest.append(some_txt.data(), some_txt.data() + some_txt.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return turbo::tlog::details::make_unique<my_formatter_flag>();
    }
};

void custom_flags_example() {

    using turbo::tlog::details::make_unique; // for pre c++14
    auto formatter = make_unique<turbo::tlog::pattern_formatter>();
    formatter->add_flag<my_formatter_flag>('*').set_pattern("[%n] [%*] [%^%l%$] %v");
    // set the new formatter using turbo::tlog::set_formatter(formatter) or logger->set_formatter(formatter)
    // turbo::tlog::set_formatter(std::move(formatter));
}


void file_events_example() {
    // pass the turbo::tlog::turbo::FileEventListener to file sinks for open/close log file notifications
    turbo::FileEventListener handlers;
    handlers.before_open = [](turbo::tlog::filename_t filename) { turbo::tlog::info("Before opening {}", filename); };
    handlers.after_open = [](turbo::tlog::filename_t filename, int fstream) {
        turbo::tlog::info("After opening {}", filename);
        std::string str =  "After opening\n";
        ::write(fstream,str.data(), str.size());
    };
    handlers.before_close = [](turbo::tlog::filename_t filename, turbo::FILE_HANDLER fd) {
        turbo::tlog::info("Before closing {}", filename);
        std::string str =  "Before closing\n";
        ::write(fd,str.data(), str.size());
    };
    handlers.after_close = [](turbo::tlog::filename_t filename) { turbo::tlog::info("After closing {}", filename); };
    auto file_sink = std::make_shared<turbo::tlog::sinks::basic_file_sink_mt>("logs/events-sample.txt", true, handlers);
    turbo::tlog::logger my_logger("some_logger", file_sink);
    my_logger.info("Some log line");
}

void replace_default_logger_example() {
    // store the old logger so we don't break other examples.
    auto old_logger = turbo::tlog::default_logger();

    auto new_logger = turbo::tlog::basic_logger_mt("new_default_logger", "logs/new-default-log.txt", true);
    turbo::tlog::set_default_logger(new_logger);
    turbo::tlog::set_level(turbo::tlog::level::info);
    turbo::tlog::debug("This message should not be displayed!");
    turbo::tlog::set_level(turbo::tlog::level::trace);
    turbo::tlog::debug("This message should be displayed..");

    turbo::tlog::set_default_logger(old_logger);
}
