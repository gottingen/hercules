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

#include "turbo/log/sinks/base_sink.h"
#include "turbo/log/details/circular_q.h"
#include "turbo/log/details/log_msg_buffer.h"
#include "turbo/log/details/null_mutex.h"

#include <mutex>
#include <string>
#include <vector>

namespace turbo::tlog {
namespace sinks {
/*
 * Ring buffer sink
 */
template<typename Mutex>
class ringbuffer_sink final : public base_sink<Mutex>
{
public:
    explicit ringbuffer_sink(size_t n_items)
        : q_{n_items}
    {}

    std::vector<details::log_msg_buffer> last_raw(size_t lim = 0)
    {
        std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
        auto items_available = q_.size();
        auto n_items = lim > 0 ? (std::min)(lim, items_available) : items_available;
        std::vector<details::log_msg_buffer> ret;
        ret.reserve(n_items);
        for (size_t i = (items_available - n_items); i < items_available; i++)
        {
            ret.push_back(q_.at(i));
        }
        return ret;
    }

    std::vector<std::string> last_formatted(size_t lim = 0)
    {
        std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
        auto items_available = q_.size();
        auto n_items = lim > 0 ? (std::min)(lim, items_available) : items_available;
        std::vector<std::string> ret;
        ret.reserve(n_items);
        for (size_t i = (items_available - n_items); i < items_available; i++)
        {
            memory_buf_t formatted;
            base_sink<Mutex>::formatter_->format(q_.at(i), formatted);
            ret.push_back(std::move(TLOG_BUF_TO_STRING(formatted)));
        }
        return ret;
    }

protected:
    void sink_it_(const details::log_msg &msg) override
    {
        q_.push_back(details::log_msg_buffer{msg});
    }
    void flush_() override {}

private:
    details::circular_q<details::log_msg_buffer> q_;
};

using ringbuffer_sink_mt = ringbuffer_sink<std::mutex>;
using ringbuffer_sink_st = ringbuffer_sink<details::null_mutex>;

} // namespace sinks

} // namespace turbo::tlog
