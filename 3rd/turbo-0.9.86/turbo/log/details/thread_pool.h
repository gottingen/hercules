// Copyright 2023 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#pragma once

#include <turbo/log/details/log_msg_buffer.h>
#include <turbo/log/details/mpmc_blocking_q.h>
#include "turbo/log/details/os.h"

#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <functional>

namespace turbo::tlog {
    class async_logger;

    namespace details {

        using async_logger_ptr = std::shared_ptr<turbo::tlog::async_logger>;

        enum class async_msg_type {
            log,
            flush,
            terminate
        };

        // Async msg to move to/from the queue
        // Movable only. should never be copied
        struct async_msg : log_msg_buffer {
            async_msg_type msg_type{async_msg_type::log};
            async_logger_ptr worker_ptr;

            async_msg() = default;

            ~async_msg() = default;

            // should only be moved in or out of the queue..
            async_msg(const async_msg &) = delete;

// support for vs2013 move
#if defined(_MSC_VER) && _MSC_VER <= 1800
            async_msg(async_msg &&other)
                : log_msg_buffer(std::move(other))
                , msg_type(other.msg_type)
                , worker_ptr(std::move(other.worker_ptr))
            {}

            async_msg &operator=(async_msg &&other)
            {
                *static_cast<log_msg_buffer *>(this) = std::move(other);
                msg_type = other.msg_type;
                worker_ptr = std::move(other.worker_ptr);
                return *this;
            }
#else // (_MSC_VER) && _MSC_VER <= 1800

            async_msg(async_msg &&) = default;

            async_msg &operator=(async_msg &&) = default;

#endif

            // construct from log_msg with given type
            async_msg(async_logger_ptr &&worker, async_msg_type the_type, const details::log_msg &m)
                    : log_msg_buffer{m}, msg_type{the_type}, worker_ptr{std::move(worker)} {}

            async_msg(async_logger_ptr &&worker, async_msg_type the_type)
                    : log_msg_buffer{}, msg_type{the_type}, worker_ptr{std::move(worker)} {}

            explicit async_msg(async_msg_type the_type)
                    : async_msg{nullptr, the_type} {}
        };

        class TURBO_DLL thread_pool {
        public:
            using item_type = async_msg;
            using q_type = details::mpmc_blocking_queue<item_type>;

            thread_pool(size_t q_max_items, size_t threads_n, std::function<void()> on_thread_start,
                        std::function<void()> on_thread_stop);

            thread_pool(size_t q_max_items, size_t threads_n, std::function<void()> on_thread_start);

            thread_pool(size_t q_max_items, size_t threads_n);

            // message all threads to terminate gracefully and join them
            ~thread_pool();

            thread_pool(const thread_pool &) = delete;

            thread_pool &operator=(thread_pool &&) = delete;

            void
            post_log(async_logger_ptr &&worker_ptr, const details::log_msg &msg, async_overflow_policy overflow_policy);

            void post_flush(async_logger_ptr &&worker_ptr, async_overflow_policy overflow_policy);

            size_t overrun_counter();

            void reset_overrun_counter();

            size_t queue_size();

        private:
            q_type q_;

            std::vector<std::thread> threads_;

            void post_async_msg_(async_msg &&new_msg, async_overflow_policy overflow_policy);

            void worker_loop_();

            // process next message in the queue
            // return true if this thread should still be active (while no terminate msg
            // was received)
            bool process_next_msg_();
        };

    } // namespace details
} // namespace turbo::tlog

