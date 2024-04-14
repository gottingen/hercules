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

#include "turbo/log/details/log_msg.h"

namespace turbo::tlog::details {

    // Extend log_msg with internal buffer to store its payload.
    // This is needed since log_msg holds string_views that points to stack data.

    class TURBO_DLL log_msg_buffer : public log_msg {
        memory_buf_t buffer;

        void update_string_views();

    public:
        log_msg_buffer() = default;

        explicit log_msg_buffer(const log_msg &orig_msg);

        log_msg_buffer(const log_msg_buffer &other);

        log_msg_buffer(log_msg_buffer &&other) noexcept;

        log_msg_buffer &operator=(const log_msg_buffer &other);

        log_msg_buffer &operator=(log_msg_buffer &&other) noexcept;
    };

} // namespace turbo::tlog::details

