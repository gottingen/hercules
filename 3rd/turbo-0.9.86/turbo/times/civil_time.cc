// Copyright 2018 The Turbo Authors.
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

#include "turbo/times/civil_time.h"

#include <cstdlib>
#include <ostream>
#include <string>
#include "turbo/status/error.h"
#include "turbo/times/time.h"
#include "turbo/format/format.h"
#include "turbo/strings/numbers.h"

namespace turbo {
    namespace {

        // Since a civil time has a larger year range than turbo::Time (64-bit years vs
        // 64-bit seconds, respectively) we normalize years to roughly +/- 400 years
        // around the year 2400, which will produce an equivalent year in a range that
        // turbo::Time can handle.
        inline civil_year_t NormalizeYear(civil_year_t year) {
            return 2400 + year % 400;
        }

        // Formats the given CivilSecond according to the given format.
        std::string FormatYearAnd(std::string_view fmt, CivilSecond cs) {
            const CivilSecond ncs(NormalizeYear(cs.year()), cs.month(), cs.day(),
                                  cs.hour(), cs.minute(), cs.second());
            const TimeZone utc = utc_time_zone();
            return format("{}{}", cs.year(), Time::from_civil(ncs, utc).to_string(fmt, utc));
        }

        template<typename CivilT>
        bool ParseYearAnd(std::string_view fmt, std::string_view s, CivilT *c) {
            // Civil times support a larger year range than turbo::Time, so we need to
            // parse the year separately, normalize it, then use turbo::parse_time on the
            // normalized string.
            const std::string ss = std::string(s);  // TODO(turbo-team): Avoid conversion.
            const char *const np = ss.c_str();
            char *endp;
            errno = 0;
            const civil_year_t y =
                    std::strtoll(np, &endp, 10);  // NOLINT(runtime/deprecated_fn)
            if (endp == np || errno == ERANGE) return false;
            const std::string norm = format("{}{}", NormalizeYear(y), endp);

            const TimeZone utc = utc_time_zone();
            Time t;
            if (t.parse_time(format("%Y{}", fmt), norm, utc, nullptr)) {
                const auto cs = to_civil_second(t, utc);
                *c = CivilT(y, cs.month(), cs.day(), cs.hour(), cs.minute(), cs.second());
                return true;
            }

            return false;
        }


        // Tries to parse the type as a CivilT1, but then assigns the result to the
        // argument of type CivilT2.
        template<typename CivilT1, typename CivilT2>
        bool ParseAs(std::string_view s, CivilT2 *c) {
            CivilT1 t1;
            if (parse_civil_time(s, &t1)) {
                *c = CivilT2(t1);
                return true;
            }
            return false;
        }

        template<typename CivilT>
        bool ParseLenient(std::string_view s, CivilT *c) {
            // A fastpath for when the given string data parses exactly into the given
            // type T (e.g., s="YYYY-MM-DD" and CivilT=CivilDay).
            if (parse_civil_time(s, c)) return true;
            // Try parsing as each of the 6 types, trying the most common types first
            // (based on csearch results).
            if (ParseAs<CivilDay>(s, c)) return true;
            if (ParseAs<CivilSecond>(s, c)) return true;
            if (ParseAs<CivilHour>(s, c)) return true;
            if (ParseAs<CivilMonth>(s, c)) return true;
            if (ParseAs<CivilMinute>(s, c)) return true;
            if (ParseAs<CivilYear>(s, c)) return true;
            return false;
        }
    }  // namespace

    std::string format_civil_time(CivilSecond c) {
        return FormatYearAnd("-%m-%d%ET%H:%M:%S", c);
    }

    std::string format_civil_time(CivilMinute c) {
        return FormatYearAnd("-%m-%d%ET%H:%M", c);
    }

    std::string format_civil_time(CivilHour c) {
        return FormatYearAnd("-%m-%d%ET%H", c);
    }

    std::string format_civil_time(CivilDay c) { return FormatYearAnd("-%m-%d", c); }

    std::string format_civil_time(CivilMonth c) { return FormatYearAnd("-%m", c); }

    std::string format_civil_time(CivilYear c) { return FormatYearAnd("", c); }

    bool parse_civil_time(std::string_view s, CivilSecond *c) {
        return ParseYearAnd("-%m-%d%ET%H:%M:%S", s, c);
    }

    bool parse_civil_time(std::string_view s, CivilMinute *c) {
        return ParseYearAnd("-%m-%d%ET%H:%M", s, c);
    }

    bool parse_civil_time(std::string_view s, CivilHour *c) {
        return ParseYearAnd("-%m-%d%ET%H", s, c);
    }

    bool parse_civil_time(std::string_view s, CivilDay *c) {
        return ParseYearAnd("-%m-%d", s, c);
    }

    bool parse_civil_time(std::string_view s, CivilMonth *c) {
        return ParseYearAnd("-%m", s, c);
    }

    bool parse_civil_time(std::string_view s, CivilYear *c) {
        return ParseYearAnd("", s, c);
    }

    bool parse_lenient_civil_time(std::string_view s, CivilSecond *c) {
        return ParseLenient(s, c);
    }

    bool parse_lenient_civil_time(std::string_view s, CivilMinute *c) {
        return ParseLenient(s, c);
    }

    bool parse_lenient_civil_time(std::string_view s, CivilHour *c) {
        return ParseLenient(s, c);
    }

    bool parse_lenient_civil_time(std::string_view s, CivilDay *c) {
        return ParseLenient(s, c);
    }

    bool parse_lenient_civil_time(std::string_view s, CivilMonth *c) {
        return ParseLenient(s, c);
    }

    bool parse_lenient_civil_time(std::string_view s, CivilYear *c) {
        return ParseLenient(s, c);
    }

    namespace time_internal {

        std::ostream &operator<<(std::ostream &os, CivilYear y) {
            return os << format_civil_time(y);
        }

        std::ostream &operator<<(std::ostream &os, CivilMonth m) {
            return os << format_civil_time(m);
        }

        std::ostream &operator<<(std::ostream &os, CivilDay d) {
            return os << format_civil_time(d);
        }

        std::ostream &operator<<(std::ostream &os, CivilHour h) {
            return os << format_civil_time(h);
        }

        std::ostream &operator<<(std::ostream &os, CivilMinute m) {
            return os << format_civil_time(m);
        }

        std::ostream &operator<<(std::ostream &os, CivilSecond s) {
            return os << format_civil_time(s);
        }

    }  // namespace time_internal

    bool turbo_parse_flag(string_view s, CivilSecond *c, std::string *) {
        return parse_lenient_civil_time(s, c);
    }

    bool turbo_parse_flag(string_view s, CivilMinute *c, std::string *) {
        return parse_lenient_civil_time(s, c);
    }

    bool turbo_parse_flag(string_view s, CivilHour *c, std::string *) {
        return parse_lenient_civil_time(s, c);
    }

    bool turbo_parse_flag(string_view s, CivilDay *c, std::string *) {
        return parse_lenient_civil_time(s, c);
    }

    bool turbo_parse_flag(string_view s, CivilMonth *c, std::string *) {
        return parse_lenient_civil_time(s, c);
    }

    bool turbo_parse_flag(string_view s, CivilYear *c, std::string *) {
        return parse_lenient_civil_time(s, c);
    }

    std::string turbo_unparse_flag(CivilSecond c) { return format_civil_time(c); }

    std::string turbo_unparse_flag(CivilMinute c) { return format_civil_time(c); }

    std::string turbo_unparse_flag(CivilHour c) { return format_civil_time(c); }

    std::string turbo_unparse_flag(CivilDay c) { return format_civil_time(c); }

    std::string turbo_unparse_flag(CivilMonth c) { return format_civil_time(c); }

    std::string turbo_unparse_flag(CivilYear c) { return format_civil_time(c); }
}  // namespace turbo
