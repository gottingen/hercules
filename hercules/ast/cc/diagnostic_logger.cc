// Copyright 2024 The titan-search Authors.
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

#include <hercules/ast/cc/diagnostic_logger.h>

#include <cstdio>
#include <mutex>

namespace hercules::ccast {

    bool diagnostic_logger::log(const char *source, const diagnostic &d) const {
        if (!verbose_ && d.severity == severity::debug)
            return false;
        return do_log(source, d);
    }

    collie::ts::object_ref<const diagnostic_logger> default_logger() noexcept {
        static const stderr_diagnostic_logger logger(false);
        return collie::ts::ref(logger);
    }

    collie::ts::object_ref<const diagnostic_logger> default_verbose_logger() noexcept {
        static const stderr_diagnostic_logger logger(true);
        return collie::ts::ref(logger);
    }

    bool stderr_diagnostic_logger::do_log(const char *source, const diagnostic &d) const {
        auto loc = d.location.to_string();
        if (loc.empty())
            std::fprintf(stderr, "[%s] [%s] %s\n", source, to_string(d.severity), d.message.c_str());
        else
            std::fprintf(stderr, "[%s] [%s] %s %s\n", source, to_string(d.severity),
                         d.location.to_string().c_str(), d.message.c_str());
        return true;
    }
}  // namespace hercules::ccast