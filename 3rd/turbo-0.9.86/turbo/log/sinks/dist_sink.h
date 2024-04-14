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

#include "base_sink.h"
#include "turbo/log/details/log_msg.h"
#include "turbo/log/details/null_mutex.h"
#include <turbo/log/pattern_formatter.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

// Distribution sink (mux). Stores a vector of sinks which get called when log
// is called

namespace turbo::tlog {
namespace sinks {

template<typename Mutex>
class dist_sink : public base_sink<Mutex>
{
public:
    dist_sink() = default;
    explicit dist_sink(std::vector<std::shared_ptr<sink>> sinks)
        : sinks_(sinks)
    {}

    dist_sink(const dist_sink &) = delete;
    dist_sink &operator=(const dist_sink &) = delete;

    void add_sink(std::shared_ptr<sink> sink)
    {
        std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
        sinks_.push_back(sink);
    }

    void remove_sink(std::shared_ptr<sink> sink)
    {
        std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
        sinks_.erase(std::remove(sinks_.begin(), sinks_.end(), sink), sinks_.end());
    }

    void set_sinks(std::vector<std::shared_ptr<sink>> sinks)
    {
        std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
        sinks_ = std::move(sinks);
    }

    std::vector<std::shared_ptr<sink>> &sinks()
    {
        return sinks_;
    }

protected:
    void sink_it_(const details::log_msg &msg) override
    {
        for (auto &sub_sink : sinks_)
        {
            if (sub_sink->should_log(msg.level))
            {
                sub_sink->log(msg);
            }
        }
    }

    void flush_() override
    {
        for (auto &sub_sink : sinks_)
        {
            sub_sink->flush();
        }
    }

    void set_pattern_(const std::string &pattern) override
    {
        set_formatter_(details::make_unique<turbo::tlog::pattern_formatter>(pattern));
    }

    void set_formatter_(std::unique_ptr<turbo::tlog::formatter> sink_formatter) override
    {
        base_sink<Mutex>::formatter_ = std::move(sink_formatter);
        for (auto &sub_sink : sinks_)
        {
            sub_sink->set_formatter(base_sink<Mutex>::formatter_->clone());
        }
    }
    std::vector<std::shared_ptr<sink>> sinks_;
};

using dist_sink_mt = dist_sink<std::mutex>;
using dist_sink_st = dist_sink<details::null_mutex>;

} // namespace sinks
} // namespace turbo::tlog
