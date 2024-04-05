// Copyright 2024 The Elastic-AI Authors.
// part of Elastic AI Search
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

#include <collie/log/common.h>
#include <collie/log/logger.h>

namespace clog {

template <typename Factory>
inline std::shared_ptr<logger> stdout_color_mt(const std::string &logger_name,
                                                      color_mode mode) {
    return Factory::template create<sinks::stdout_color_sink_mt>(logger_name, mode);
}

template <typename Factory>
inline std::shared_ptr<logger> stdout_color_st(const std::string &logger_name,
                                                      color_mode mode) {
    return Factory::template create<sinks::stdout_color_sink_st>(logger_name, mode);
}

template <typename Factory>
inline std::shared_ptr<logger> stderr_color_mt(const std::string &logger_name,
                                                      color_mode mode) {
    return Factory::template create<sinks::stderr_color_sink_mt>(logger_name, mode);
}

template <typename Factory>
inline std::shared_ptr<logger> stderr_color_st(const std::string &logger_name,
                                                      color_mode mode) {
    return Factory::template create<sinks::stderr_color_sink_st>(logger_name, mode);
}
}  // namespace clog
