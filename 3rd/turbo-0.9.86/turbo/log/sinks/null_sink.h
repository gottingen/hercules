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

#include "turbo/log/details/null_mutex.h"
#include <turbo/log/sinks/base_sink.h>
#include "turbo/log/details/synchronous_factory.h"

#include <mutex>

namespace turbo::tlog {
namespace sinks {

template<typename Mutex>
class null_sink : public base_sink<Mutex>
{
protected:
    void sink_it_(const details::log_msg &) override {}
    void flush_() override {}
};

using null_sink_mt = null_sink<details::null_mutex>;
using null_sink_st = null_sink<details::null_mutex>;

} // namespace sinks

template<typename Factory = turbo::tlog::synchronous_factory>
inline std::shared_ptr<logger> null_logger_mt(const std::string &logger_name)
{
    auto null_logger = Factory::template create<sinks::null_sink_mt>(logger_name);
    null_logger->set_level(level::off);
    return null_logger;
}

template<typename Factory = turbo::tlog::synchronous_factory>
inline std::shared_ptr<logger> null_logger_st(const std::string &logger_name)
{
    auto null_logger = Factory::template create<sinks::null_sink_st>(logger_name);
    null_logger->set_level(level::off);
    return null_logger;
}

} // namespace turbo::tlog
