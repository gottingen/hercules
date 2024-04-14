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

#include "registry.h"

namespace turbo::tlog {

    // Default logger factory-  creates synchronous loggers
    class logger;

    struct synchronous_factory {
        template<typename Sink, typename... SinkArgs>
        static std::shared_ptr<turbo::tlog::logger> create(std::string logger_name, SinkArgs &&... args) {
            auto sink = std::make_shared<Sink>(std::forward<SinkArgs>(args)...);
            auto new_logger = std::make_shared<turbo::tlog::logger>(std::move(logger_name), std::move(sink));
            details::registry::instance().initialize_logger(new_logger);
            return new_logger;
        }
    };
} // namespace turbo::tlog
