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
#include "turbo/log/formatter.h"

namespace turbo::tlog {

    namespace sinks {
        class TURBO_DLL sink {
        public:
            virtual ~sink() = default;

            virtual void log(const details::log_msg &msg) = 0;

            virtual void flush() = 0;

            virtual void set_pattern(const std::string &pattern) = 0;

            virtual void set_formatter(std::unique_ptr<turbo::tlog::formatter> sink_formatter) = 0;

            void set_level(level::level_enum log_level);

            level::level_enum level() const;

            bool should_log(level::level_enum msg_level) const;

        protected:
            // sink log level - default is all
            level_t level_{level::trace};
        };

    } // namespace sinks
} // namespace turbo::tlog

