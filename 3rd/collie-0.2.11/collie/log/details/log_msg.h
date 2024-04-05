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
#include <string>

namespace clog {
    namespace details {
        struct log_msg {
            log_msg() = default;

            log_msg(log_clock::time_point log_time,
                    source_loc loc,
                    string_view_t logger_name,
                    level::level_enum lvl,
                    string_view_t msg);

            log_msg(source_loc loc, string_view_t logger_name, level::level_enum lvl, string_view_t msg);

            log_msg(string_view_t logger_name, level::level_enum lvl, string_view_t msg);

            log_msg(const log_msg &other) = default;

            log_msg &operator=(const log_msg &other) = default;

            string_view_t logger_name;
            level::level_enum level{level::off};
            log_clock::time_point time;
            size_t thread_id{0};

            // wrapping the formatted text with color (updated by pattern_formatter).
            mutable size_t color_range_start{0};
            mutable size_t color_range_end{0};

            source_loc source;
            string_view_t payload;
        };
    }  // namespace details
}  // namespace clog

#include <collie/log/details/log_msg-inl.h>
