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

#include <collie/log/details/registry.h>

namespace clog {

    // Default logger factory-  creates synchronous loggers
    class logger;

    struct synchronous_factory {
        template<typename Sink, typename... SinkArgs>
        static std::shared_ptr<clog::logger> create(std::string logger_name, SinkArgs &&...args) {
            auto sink = std::make_shared<Sink>(std::forward<SinkArgs>(args)...);
            auto new_logger = std::make_shared<clog::logger>(std::move(logger_name), std::move(sink));
            details::registry::instance().initialize_logger(new_logger);
            return new_logger;
        }
    };
}  // namespace clog
