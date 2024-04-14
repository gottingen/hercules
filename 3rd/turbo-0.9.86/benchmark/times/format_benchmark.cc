// Copyright 2018 The Turbo Authors.
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

#include <cstddef>
#include <string>

#include "turbo/times/time.h"
#include "benchmark/benchmark.h"
#include "turbo/base/internal/raw_logging.h"

namespace turbo {

    namespace time_internal {

        TimeZone load_time_zone(const std::string &name) {
            TimeZone tz;
            TURBO_RAW_CHECK(load_time_zone(name, &tz), name.c_str());
            return tz;
        }

    }  // namespace time_internal

}  // namespace turbo

namespace {

    namespace {
        const char *const kFormats[] = {
                turbo::RFC1123_full,     // 0
                turbo::RFC1123_no_wday,  // 1
                turbo::RFC3339_full,     // 2
                turbo::RFC3339_sec,      // 3
                "%Y-%m-%d%ET%H:%M:%S",  // 4
                "%Y-%m-%d",             // 5
        };
        const int kNumFormats = sizeof(kFormats) / sizeof(kFormats[0]);
    }  // namespace

    void BM_Format_FormatTime(benchmark::State &state) {
        const std::string fmt = kFormats[state.range(0)];
        state.SetLabel(fmt);
        const turbo::TimeZone lax =
                turbo::time_internal::load_time_zone("America/Los_Angeles");
        const turbo::Time t =
                turbo::Time::from_civil(turbo::CivilSecond(1977, 6, 28, 9, 8, 7), lax) +
                turbo::Duration::nanoseconds(1);
        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(t.to_string(fmt, lax).length());
        }
    }

    BENCHMARK(BM_Format_FormatTime)->DenseRange(0, kNumFormats - 1);

    void BM_Format_ParseTime(benchmark::State &state) {
        const std::string fmt = kFormats[state.range(0)];
        state.SetLabel(fmt);
        const turbo::TimeZone lax =
                turbo::time_internal::load_time_zone("America/Los_Angeles");
        turbo::Time t = turbo::Time::from_civil(turbo::CivilSecond(1977, 6, 28, 9, 8, 7), lax) +
                        turbo::Duration::nanoseconds(1);
        const std::string when = t.to_string(fmt, lax);
        std::string err;
        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(t.parse_time(fmt, when, lax, &err));
        }
    }

    BENCHMARK(BM_Format_ParseTime)->DenseRange(0, kNumFormats - 1);

}  // namespace
