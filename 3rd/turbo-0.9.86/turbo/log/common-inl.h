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
#include <algorithm>
#include <iterator>

namespace turbo::tlog {
    namespace level {

        constexpr static std::string_view level_string_views[]TLOG_LEVEL_NAMES;

        static const char *short_level_names[]TLOG_SHORT_LEVEL_NAMES;

        const std::string_view &to_string_view(turbo::tlog::level::level_enum l) noexcept {
            return level_string_views[l];
        }

        const char *to_short_c_str(turbo::tlog::level::level_enum l) noexcept {
            return short_level_names[l];
        }

        turbo::tlog::level::level_enum from_str(const std::string &name) noexcept {
            auto it = std::find(std::begin(level_string_views), std::end(level_string_views), name);
            if (it != std::end(level_string_views))
                return static_cast<level::level_enum>(std::distance(std::begin(level_string_views), it));

            // check also for "warn" and "err" before giving up..
            if (name == "warn") {
                return level::warn;
            }
            if (name == "err") {
                return level::err;
            }
            return level::off;
        }
    } // namespace level

    tlog_ex::tlog_ex(std::string msg)
            : msg_(std::move(msg)) {}

    tlog_ex::tlog_ex(const std::string &msg, int last_errno) {
        memory_buf_t outbuf;
        turbo::format_system_error(outbuf, last_errno, msg.c_str());
        msg_ = turbo::to_string(outbuf);
    }

    const char *tlog_ex::what() const noexcept {
        return msg_.c_str();
    }

    void throw_tlog_ex(const std::string &msg, int last_errno) {
        TLOG_THROW(tlog_ex(msg, last_errno));
    }

    void throw_tlog_ex(std::string msg) {
        TLOG_THROW(tlog_ex(std::move(msg)));
    }

} // namespace turbo::tlog
