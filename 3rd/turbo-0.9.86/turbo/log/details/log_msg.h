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

#include "turbo/log/common.h"
#include <string>

namespace turbo::tlog::details {
    struct TURBO_DLL log_msg {
        log_msg() = default;

        log_msg(turbo::Time log_time, source_loc loc, std::string_view logger_name, level::level_enum lvl,
                std::string_view msg);

        log_msg(source_loc loc, std::string_view logger_name, level::level_enum lvl, std::string_view msg);

        log_msg(std::string_view logger_name, level::level_enum lvl, std::string_view msg);

        log_msg(const log_msg &other) = default;

        log_msg &operator=(const log_msg &other) = default;

        std::string_view logger_name;
        level::level_enum level{level::off};
        turbo::Time time;
        size_t thread_id{0};

        // wrapping the formatted text with color (updated by pattern_formatter).
        mutable size_t color_range_start{0};
        mutable size_t color_range_end{0};

        source_loc source;
        std::string_view payload;
    };
} // namespace turbo::tlog::details

