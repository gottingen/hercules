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

#include <turbo/log/sinks/stdout_color_sinks.h>
#include "turbo/log/logger.h"
#include "turbo/log/common.h"

namespace turbo::tlog {

template<typename Factory>
 std::shared_ptr<logger> stdout_color_mt(const std::string &logger_name, color_mode mode)
{
    return Factory::template create<sinks::stdout_color_sink_mt>(logger_name, mode);
}

template<typename Factory>
 std::shared_ptr<logger> stdout_color_st(const std::string &logger_name, color_mode mode)
{
    return Factory::template create<sinks::stdout_color_sink_st>(logger_name, mode);
}

template<typename Factory>
 std::shared_ptr<logger> stderr_color_mt(const std::string &logger_name, color_mode mode)
{
    return Factory::template create<sinks::stderr_color_sink_mt>(logger_name, mode);
}

template<typename Factory>
 std::shared_ptr<logger> stderr_color_st(const std::string &logger_name, color_mode mode)
{
    return Factory::template create<sinks::stderr_color_sink_st>(logger_name, mode);
}
} // namespace turbo::tlog
