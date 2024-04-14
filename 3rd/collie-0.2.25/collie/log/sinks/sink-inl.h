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

inline bool clog::sinks::sink::should_log(clog::level::level_enum msg_level) const {
    return msg_level >= level_.load(std::memory_order_relaxed);
}

inline void clog::sinks::sink::set_level(level::level_enum log_level) {
    level_.store(log_level, std::memory_order_relaxed);
}

inline clog::level::level_enum clog::sinks::sink::level() const {
    return static_cast<clog::level::level_enum>(level_.load(std::memory_order_relaxed));
}
