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

#include <turbo/log/sinks/sink.h>

#include "turbo/log/common.h"

bool turbo::tlog::sinks::sink::should_log(turbo::tlog::level::level_enum msg_level) const {
    return msg_level >= level_.load(std::memory_order_relaxed);
}

void turbo::tlog::sinks::sink::set_level(level::level_enum log_level) {
    level_.store(log_level, std::memory_order_relaxed);
}

turbo::tlog::level::level_enum turbo::tlog::sinks::sink::level() const {
    return static_cast<turbo::tlog::level::level_enum>(level_.load(std::memory_order_relaxed));
}
