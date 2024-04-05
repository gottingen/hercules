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

#include <algorithm>
#include <iterator>

namespace clog {
    namespace level {

        constexpr static string_view_t level_string_views[]CLOG_LEVEL_NAMES;

        static const char *short_level_names[]CLOG_SHORT_LEVEL_NAMES;

        inline const string_view_t &to_string_view(clog::level::level_enum l) noexcept {
            return level_string_views[l];
        }

        inline const char *to_short_c_str(clog::level::level_enum l) noexcept {
            return short_level_names[l];
        }

        inline clog::level::level_enum from_str(const std::string &name) noexcept {
            auto it = std::find(std::begin(level_string_views), std::end(level_string_views), name);
            if (it != std::end(level_string_views))
                return static_cast<level::level_enum>(std::distance(std::begin(level_string_views), it));

            // check also for "warn" and "err" before giving up..
            if (name == "warn") {
                return level::warn;
            }
            if (name == "err") {
                return level::error;
            }
            return level::off;
        }
    }  // namespace level

    inline CLogEx::CLogEx(std::string msg)
            : msg_(std::move(msg)) {}

    inline CLogEx::CLogEx(const std::string &msg, int last_errno) {
        memory_buf_t outbuf;
        fmt::format_system_error(outbuf, last_errno, msg.c_str());
        msg_ = fmt::to_string(outbuf);
    }

    inline const char *CLogEx::what() const noexcept { return msg_.c_str(); }

    inline void throw_clog_ex(const std::string &msg, int last_errno) {
        CLOG_THROW(CLogEx(msg, last_errno));
    }

    inline void throw_clog_ex(std::string msg) { CLOG_THROW(CLogEx(std::move(msg))); }

}  // namespace clog
