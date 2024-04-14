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

#include <turbo/log/sinks/base_sink.h>
#include "turbo/log/common.h"
#include <turbo/log/pattern_formatter.h>

#include <memory>

template<typename Mutex>
 turbo::tlog::sinks::base_sink<Mutex>::base_sink()
    : formatter_{details::make_unique<turbo::tlog::pattern_formatter>()}
{}

template<typename Mutex>
 turbo::tlog::sinks::base_sink<Mutex>::base_sink(std::unique_ptr<turbo::tlog::formatter> formatter)
    : formatter_{std::move(formatter)}
{}

template<typename Mutex>
void  turbo::tlog::sinks::base_sink<Mutex>::log(const details::log_msg &msg)
{
    std::lock_guard<Mutex> lock(mutex_);
    sink_it_(msg);
}

template<typename Mutex>
void  turbo::tlog::sinks::base_sink<Mutex>::flush()
{
    std::lock_guard<Mutex> lock(mutex_);
    flush_();
}

template<typename Mutex>
void  turbo::tlog::sinks::base_sink<Mutex>::set_pattern(const std::string &pattern)
{
    std::lock_guard<Mutex> lock(mutex_);
    set_pattern_(pattern);
}

template<typename Mutex>
void  turbo::tlog::sinks::base_sink<Mutex>::set_formatter(std::unique_ptr<turbo::tlog::formatter> sink_formatter)
{
    std::lock_guard<Mutex> lock(mutex_);
    set_formatter_(std::move(sink_formatter));
}

template<typename Mutex>
void  turbo::tlog::sinks::base_sink<Mutex>::set_pattern_(const std::string &pattern)
{
    set_formatter_(details::make_unique<turbo::tlog::pattern_formatter>(pattern));
}

template<typename Mutex>
void  turbo::tlog::sinks::base_sink<Mutex>::set_formatter_(std::unique_ptr<turbo::tlog::formatter> sink_formatter)
{
    formatter_ = std::move(sink_formatter);
}
