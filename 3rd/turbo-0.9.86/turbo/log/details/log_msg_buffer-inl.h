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

#include <turbo/log/details/log_msg_buffer.h>

namespace turbo::tlog::details {

    log_msg_buffer::log_msg_buffer(const log_msg &orig_msg)
            : log_msg{orig_msg} {
        buffer.append(logger_name.begin(), logger_name.end());
        buffer.append(payload.begin(), payload.end());
        update_string_views();
    }

    log_msg_buffer::log_msg_buffer(const log_msg_buffer &other)
            : log_msg{other} {
        buffer.append(logger_name.begin(), logger_name.end());
        buffer.append(payload.begin(), payload.end());
        update_string_views();
    }

    log_msg_buffer::log_msg_buffer(log_msg_buffer &&other) noexcept: log_msg{other}, buffer{std::move(other.buffer)} {
        update_string_views();
    }

    log_msg_buffer &log_msg_buffer::operator=(const log_msg_buffer &other) {
        log_msg::operator=(other);
        buffer.clear();
        buffer.append(other.buffer.data(), other.buffer.data() + other.buffer.size());
        update_string_views();
        return *this;
    }

    log_msg_buffer &log_msg_buffer::operator=(log_msg_buffer &&other) noexcept {
        log_msg::operator=(other);
        buffer = std::move(other.buffer);
        update_string_views();
        return *this;
    }

    void log_msg_buffer::update_string_views() {
        logger_name = std::string_view{buffer.data(), logger_name.size()};
        payload = std::string_view{buffer.data() + logger_name.size(), payload.size()};
    }

} // namespace turbo::tlog::details

