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

#ifndef TURBO_BASE_LOG_SEVERITY_H_
#define TURBO_BASE_LOG_SEVERITY_H_

#include <array>
#include <ostream>

#include "turbo/platform/port.h"

namespace turbo {

    // turbo::LogSeverity
    //
    // Four severity levels are defined. Logging APIs should terminate the program
    // when a message is logged at severity `kFatal`; the other levels have no
    // special semantics.
    //
    // Values other than the four defined levels (e.g. produced by `static_cast`)
    // are valid, but their semantics when passed to a function, macro, or flag
    // depend on the function, macro, or flag. The usual behavior is to normalize
    // such values to a defined severity level, however in some cases values other
    // than the defined levels are useful for comparison.
    //
    // Example:
    //
    //   // Effectively disables all logging:
    //   SetMinLogLevel(static_cast<turbo::LogSeverity>(100));
    //
    // overload is defined there as well for consistency.
    //
    // turbo::LogSeverity Flag String Representation
    //
    // An `turbo::LogSeverity` has a string representation used for parsing
    // command-line flags based on the enumerator name (e.g. `kFatal`) or
    // its unprefixed name (without the `k`) in any case-insensitive form. (E.g.
    // "FATAL", "fatal" or "Fatal" are all valid.) Unparsing such flags produces an
    // unprefixed string representation in all caps (e.g. "FATAL") or an integer.
    //
    // Additionally, the parser accepts arbitrary integers (as if the type were
    // `int`).
    //
    // Examples:
    //
    //   --my_log_level=kInfo
    //   --my_log_level=INFO
    //   --my_log_level=info
    //   --my_log_level=0
    //
    // Unparsing a flag produces the same result as `turbo::LogSeverityName()` for
    // the standard levels and a base-ten integer otherwise.
    enum class LogSeverity : int {
        kInfo = 0,
        kWarning = 1,
        kError = 2,
        kFatal = 3,
    };

    // LogSeverities()
    //
    // Returns an iterable of all standard `turbo::LogSeverity` values, ordered from
    // least to most severe.
    constexpr std::array<turbo::LogSeverity, 4> LogSeverities() {
        return {{turbo::LogSeverity::kInfo, turbo::LogSeverity::kWarning,
                 turbo::LogSeverity::kError, turbo::LogSeverity::kFatal}};
    }

    // LogSeverityName()
    //
    // Returns the all-caps string representation (e.g. "INFO") of the specified
    // severity level if it is one of the standard levels and "UNKNOWN" otherwise.
    constexpr const char *LogSeverityName(turbo::LogSeverity s) {
        return s == turbo::LogSeverity::kInfo
               ? "INFO"
               : s == turbo::LogSeverity::kWarning
                 ? "WARNING"
                 : s == turbo::LogSeverity::kError
                   ? "ERROR"
                   : s == turbo::LogSeverity::kFatal ? "FATAL" : "UNKNOWN";
    }

    // NormalizeLogSeverity()
    //
    // Values less than `kInfo` normalize to `kInfo`; values greater than `kFatal`
    // normalize to `kError` (**NOT** `kFatal`).
    constexpr turbo::LogSeverity NormalizeLogSeverity(turbo::LogSeverity s) {
        return s < turbo::LogSeverity::kInfo
               ? turbo::LogSeverity::kInfo
               : s > turbo::LogSeverity::kFatal ? turbo::LogSeverity::kError : s;
    }

    constexpr turbo::LogSeverity NormalizeLogSeverity(int s) {
        return turbo::NormalizeLogSeverity(static_cast<turbo::LogSeverity>(s));
    }

    // operator<<
    //
    // The exact representation of a streamed `turbo::LogSeverity` is deliberately
    // unspecified; do not rely on it.
    std::ostream &operator<<(std::ostream &os, turbo::LogSeverity s);

    // Enums representing a lower bound for LogSeverity. APIs that only operate on
    // messages of at least a certain level (for example, `SetMinLogLevel()`) use
    // this type to specify that level. turbo::LogSeverityAtLeast::kInfinity is
    // a level above all threshold levels and therefore no log message will
    // ever meet this threshold.
    enum class LogSeverityAtLeast : int {
        kInfo = static_cast<int>(turbo::LogSeverity::kInfo),
        kWarning = static_cast<int>(turbo::LogSeverity::kWarning),
        kError = static_cast<int>(turbo::LogSeverity::kError),
        kFatal = static_cast<int>(turbo::LogSeverity::kFatal),
        kInfinity = 1000,
    };

    std::ostream &operator<<(std::ostream &os, turbo::LogSeverityAtLeast s);

    // Enums representing an upper bound for LogSeverity. APIs that only operate on
    // messages of at most a certain level (for example, buffer all messages at or
    // below a certain level) use this type to specify that level.
    // turbo::LogSeverityAtMost::kNegativeInfinity is a level below all threshold
    // levels and therefore will exclude all log messages.
    enum class LogSeverityAtMost : int {
        kNegativeInfinity = -1000,
        kInfo = static_cast<int>(turbo::LogSeverity::kInfo),
        kWarning = static_cast<int>(turbo::LogSeverity::kWarning),
        kError = static_cast<int>(turbo::LogSeverity::kError),
        kFatal = static_cast<int>(turbo::LogSeverity::kFatal),
    };

    std::ostream &operator<<(std::ostream &os, turbo::LogSeverityAtMost s);

#define COMPOP(op1, op2, T)                                         \
  constexpr bool operator op1(turbo::T lhs, turbo::LogSeverity rhs) { \
    return static_cast<turbo::LogSeverity>(lhs) op1 rhs;             \
  }                                                                 \
  constexpr bool operator op2(turbo::LogSeverity lhs, turbo::T rhs) { \
    return lhs op2 static_cast<turbo::LogSeverity>(rhs);             \
  }

    // Comparisons between `LogSeverity` and `LogSeverityAtLeast`/
    // `LogSeverityAtMost` are only supported in one direction.
    // Valid checks are:
    //   LogSeverity >= LogSeverityAtLeast
    //   LogSeverity < LogSeverityAtLeast
    //   LogSeverity <= LogSeverityAtMost
    //   LogSeverity > LogSeverityAtMost
    COMPOP(>, <, LogSeverityAtLeast)

    COMPOP(<=, >=, LogSeverityAtLeast)

    COMPOP(<,>, LogSeverityAtMost)

    COMPOP(>=, <=, LogSeverityAtMost)

#undef COMPOP

}  // namespace turbo

#endif  // TURBO_BASE_LOG_SEVERITY_H_
