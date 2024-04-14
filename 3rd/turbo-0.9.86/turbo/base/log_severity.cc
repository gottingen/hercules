// Copyright 2020 The Turbo Authors.
//
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

#include "log_severity.h"

#include <ostream>

#include "turbo/platform/port.h"

namespace turbo {

    std::ostream &operator<<(std::ostream &os, turbo::LogSeverity s) {
        if (s == turbo::NormalizeLogSeverity(s)) return os << turbo::LogSeverityName(s);
        return os << "turbo::LogSeverity(" << static_cast<int>(s) << ")";
    }

    std::ostream &operator<<(std::ostream &os, turbo::LogSeverityAtLeast s) {
        switch (s) {
            case turbo::LogSeverityAtLeast::kInfo:
            case turbo::LogSeverityAtLeast::kWarning:
            case turbo::LogSeverityAtLeast::kError:
            case turbo::LogSeverityAtLeast::kFatal:
                return os << ">=" << static_cast<turbo::LogSeverity>(s);
            case turbo::LogSeverityAtLeast::kInfinity:
                return os << "INFINITY";
        }
        return os;
    }

    std::ostream &operator<<(std::ostream &os, turbo::LogSeverityAtMost s) {
        switch (s) {
            case turbo::LogSeverityAtMost::kInfo:
            case turbo::LogSeverityAtMost::kWarning:
            case turbo::LogSeverityAtMost::kError:
            case turbo::LogSeverityAtMost::kFatal:
                return os << "<=" << static_cast<turbo::LogSeverity>(s);
            case turbo::LogSeverityAtMost::kNegativeInfinity:
                return os << "NEGATIVE_INFINITY";
        }
        return os;
    }

}  // namespace turbo
