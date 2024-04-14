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

#include <turbo/log/details/periodic_worker.h>

namespace turbo::tlog::details {

    // stop the worker thread and join it
    periodic_worker::~periodic_worker() {
        if (worker_thread_.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                active_ = false;
            }
            cv_.notify_one();
            worker_thread_.join();
        }
    }

} // namespace turbo::tlog::details
