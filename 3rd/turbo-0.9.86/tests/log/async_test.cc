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

#define TEST_FILENAME "test_logs/async_test.log"

TEST_CASE("basic async test [async]")
{
    auto test_sink = std::make_shared<turbo::tlog::sinks::test_sink_mt>();
    size_t overrun_counter = 0;
    size_t queue_size = 128;
    size_t messages = 256;
    {
        auto tp = std::make_shared<turbo::tlog::details::thread_pool>(queue_size, 1);
        auto logger = std::make_shared<turbo::tlog::async_logger>("as", test_sink, tp, turbo::tlog::async_overflow_policy::block);
        for (size_t i = 0; i < messages; i++)
        {
            logger->info("Hello message #{}", i);
        }
        logger->flush();
        overrun_counter = tp->overrun_counter();
    }
    REQUIRE(test_sink->msg_counter() == messages);
    REQUIRE(test_sink->flush_counter() == 1);
    REQUIRE(overrun_counter == 0);
}

TEST_CASE("discard policy [async]")
{
    auto test_sink = std::make_shared<turbo::tlog::sinks::test_sink_mt>();
    test_sink->set_delay(std::chrono::milliseconds(1));
    size_t queue_size = 4;
    size_t messages = 1024;

    auto tp = std::make_shared<turbo::tlog::details::thread_pool>(queue_size, 1);
    auto logger = std::make_shared<turbo::tlog::async_logger>("as", test_sink, tp, turbo::tlog::async_overflow_policy::overrun_oldest);
    for (size_t i = 0; i < messages; i++)
    {
        logger->info("Hello message");
    }
    REQUIRE(test_sink->msg_counter() < messages);
    REQUIRE(tp->overrun_counter() > 0);
}

TEST_CASE("discard policy using factory [async]")
{
    size_t queue_size = 4;
    size_t messages = 1024;
    turbo::tlog::init_thread_pool(queue_size, 1);

    auto logger = turbo::tlog::create_async_nb<turbo::tlog::sinks::test_sink_mt>("as2");
    auto test_sink = std::static_pointer_cast<turbo::tlog::sinks::test_sink_mt>(logger->sinks()[0]);
    test_sink->set_delay(std::chrono::milliseconds(3));

    for (size_t i = 0; i < messages; i++)
    {
        logger->info("Hello message");
    }

    REQUIRE(test_sink->msg_counter() < messages);
    turbo::tlog::drop_all();
}

TEST_CASE("flush [async]")
{
    auto test_sink = std::make_shared<turbo::tlog::sinks::test_sink_mt>();
    size_t queue_size = 256;
    size_t messages = 256;
    {
        auto tp = std::make_shared<turbo::tlog::details::thread_pool>(queue_size, 1);
        auto logger = std::make_shared<turbo::tlog::async_logger>("as", test_sink, tp, turbo::tlog::async_overflow_policy::block);
        for (size_t i = 0; i < messages; i++)
        {
            logger->info("Hello message #{}", i);
        }

        logger->flush();
    }
    // std::this_thread::sleep_for(std::chrono::milliseconds(250));
    REQUIRE(test_sink->msg_counter() == messages);
    REQUIRE(test_sink->flush_counter() == 1);
}

TEST_CASE("async periodic flush [async]")
{

    auto logger = turbo::tlog::create_async<turbo::tlog::sinks::test_sink_mt>("as");
    auto test_sink = std::static_pointer_cast<turbo::tlog::sinks::test_sink_mt>(logger->sinks()[0]);

    turbo::tlog::flush_every(std::chrono::seconds(1));
    std::this_thread::sleep_for(std::chrono::milliseconds(1700));
    REQUIRE(test_sink->flush_counter() == 1);
    turbo::tlog::flush_every(std::chrono::seconds(0));
    turbo::tlog::drop_all();
}

TEST_CASE("tp->wait_empty()  [async]")
{
    auto test_sink = std::make_shared<turbo::tlog::sinks::test_sink_mt>();
    test_sink->set_delay(std::chrono::milliseconds(5));
    size_t messages = 100;

    auto tp = std::make_shared<turbo::tlog::details::thread_pool>(messages, 2);
    auto logger = std::make_shared<turbo::tlog::async_logger>("as", test_sink, tp, turbo::tlog::async_overflow_policy::block);
    for (size_t i = 0; i < messages; i++)
    {
        logger->info("Hello message #{}", i);
    }
    logger->flush();
    tp.reset();

    REQUIRE(test_sink->msg_counter() == messages);
    REQUIRE(test_sink->flush_counter() == 1);
}

TEST_CASE("multi threads [async]")
{
    auto test_sink = std::make_shared<turbo::tlog::sinks::test_sink_mt>();
    size_t queue_size = 128;
    size_t messages = 256;
    size_t n_threads = 10;
    {
        auto tp = std::make_shared<turbo::tlog::details::thread_pool>(queue_size, 1);
        auto logger = std::make_shared<turbo::tlog::async_logger>("as", test_sink, tp, turbo::tlog::async_overflow_policy::block);

        std::vector<std::thread> threads;
        for (size_t i = 0; i < n_threads; i++)
        {
            threads.emplace_back([logger, messages] {
                for (size_t j = 0; j < messages; j++)
                {
                    logger->info("Hello message #{}", j);
                }
            });
            logger->flush();
        }

        for (auto &t : threads)
        {
            t.join();
        }
    }

    REQUIRE(test_sink->msg_counter() == messages * n_threads);
    REQUIRE(test_sink->flush_counter() == n_threads);
}

TEST_CASE("to_file [async]")
{
    prepare_logdir();
    size_t messages = 1024;
    size_t tp_threads = 1;
    turbo::tlog::filename_t filename = TLOG_FILENAME_T(TEST_FILENAME);
    {
        auto file_sink = std::make_shared<turbo::tlog::sinks::basic_file_sink_mt>(filename, true);
        auto tp = std::make_shared<turbo::tlog::details::thread_pool>(messages, tp_threads);
        auto logger = std::make_shared<turbo::tlog::async_logger>("as", std::move(file_sink), std::move(tp));

        for (size_t j = 0; j < messages; j++)
        {
            logger->info("Hello message #{}", j);
        }
    }

    require_message_count(TEST_FILENAME, messages);
    auto contents = file_contents(TEST_FILENAME);
    using turbo::tlog::details::os::default_eol;
    REQUIRE(ends_with(contents, turbo::format("Hello message #1023{}", default_eol)));
}

TEST_CASE("to_file multi-workers [async]")
{
    prepare_logdir();
    size_t messages = 1024 * 10;
    size_t tp_threads = 10;
    turbo::tlog::filename_t filename = TLOG_FILENAME_T(TEST_FILENAME);
    {
        auto file_sink = std::make_shared<turbo::tlog::sinks::basic_file_sink_mt>(filename, true);
        auto tp = std::make_shared<turbo::tlog::details::thread_pool>(messages, tp_threads);
        auto logger = std::make_shared<turbo::tlog::async_logger>("as", std::move(file_sink), std::move(tp));

        for (size_t j = 0; j < messages; j++)
        {
            logger->info("Hello message #{}", j);
        }
    }

    require_message_count(TEST_FILENAME, messages);
}
