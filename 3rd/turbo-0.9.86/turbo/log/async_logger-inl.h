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

#include "turbo/log/async_logger.h"
#include <turbo/log/sinks/sink.h>
#include "turbo/log/details/thread_pool.h"

#include <memory>
#include <string>

turbo::tlog::async_logger::async_logger(
        std::string logger_name, sinks_init_list sinks_list, std::weak_ptr<details::thread_pool> tp,
        async_overflow_policy overflow_policy)
        : async_logger(std::move(logger_name), sinks_list.begin(), sinks_list.end(), std::move(tp), overflow_policy) {}

turbo::tlog::async_logger::async_logger(
        std::string logger_name, sink_ptr single_sink, std::weak_ptr<details::thread_pool> tp,
        async_overflow_policy overflow_policy)
        : async_logger(std::move(logger_name), {std::move(single_sink)}, std::move(tp), overflow_policy) {}

// send the log message to the thread pool
void turbo::tlog::async_logger::sink_it_(const details::log_msg &msg) {
    if (auto pool_ptr = thread_pool_.lock()) {
        pool_ptr->post_log(shared_from_this(), msg, overflow_policy_);
    } else {
        throw_tlog_ex("async log: thread pool doesn't exist anymore");
    }
}

// send flush request to the thread pool
void turbo::tlog::async_logger::flush_() {
    if (auto pool_ptr = thread_pool_.lock()) {
        pool_ptr->post_flush(shared_from_this(), overflow_policy_);
    } else {
        throw_tlog_ex("async flush: thread pool doesn't exist anymore");
    }
}

//
// backend functions - called from the thread pool to do the actual job
//
void turbo::tlog::async_logger::backend_sink_it_(const details::log_msg &msg) {
    for (auto &sink: sinks_) {
        if (sink->should_log(msg.level)) {
            TLOG_TRY {
                sink->log(msg);
            }
            TLOG_LOGGER_CATCH(msg.source)
        }
    }

    if (should_flush_(msg)) {
        backend_flush_();
    }
}

void turbo::tlog::async_logger::backend_flush_() {
    for (auto &sink: sinks_) {
        TLOG_TRY {
            sink->flush();
        }
        TLOG_LOGGER_CATCH(source_loc())
    }
}

std::shared_ptr<turbo::tlog::logger> turbo::tlog::async_logger::clone(std::string new_name) {
    auto cloned = std::make_shared<turbo::tlog::async_logger>(*this);
    cloned->name_ = std::move(new_name);
    return cloned;
}
