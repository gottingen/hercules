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

#include <numeric>
#include <vector>

#include "turbo/hash/hash.h"
#include "benchmark/benchmark.h"

namespace {

    // Run on (12 X 3492 MHz CPUs); 2018-11-05T13:44:29.814239103-08:00
    // CPU: Intel Haswell with HyperThreading (6 cores) dL1:32KB dL2:256KB dL3:15MB
    // Benchmark                 Time(ns)        CPU(ns)     Iterations
    // ----------------------------------------------------------------
    // BM_Difference_Days              14.5           14.5     48531105
    // BM_Step_Days                    12.6           12.6     54876006
    // BM_Format                      587            587        1000000
    // BM_Parse                       692            692        1000000
    // BM_RoundTripFormatParse       1309           1309         532075
    // BM_CivilYearTurboHash             0.710          0.710  976400000
    // BM_CivilMonthTurboHash            1.13           1.13   619500000
    // BM_CivilDayTurboHash              1.70           1.70   426000000
    // BM_CivilHourTurboHash             2.45           2.45   287600000
    // BM_CivilMinuteTurboHash           3.21           3.21   226200000
    // BM_CivilSecondTurboHash           4.10           4.10   171800000

    void BM_Difference_Days(benchmark::State &state) {
        const turbo::CivilDay c(2014, 8, 22);
        const turbo::CivilDay epoch(1970, 1, 1);
        while (state.KeepRunning()) {
            const turbo::civil_diff_t n = c - epoch;
            benchmark::DoNotOptimize(n);
        }
    }

    BENCHMARK(BM_Difference_Days);

    void BM_Step_Days(benchmark::State &state) {
        const turbo::CivilDay kStart(2014, 8, 22);
        turbo::CivilDay c = kStart;
        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(++c);
        }
    }

    BENCHMARK(BM_Step_Days);

    void BM_Format(benchmark::State &state) {
        const turbo::CivilSecond c(2014, 1, 2, 3, 4, 5);
        while (state.KeepRunning()) {
            const std::string s = turbo::format_civil_time(c);
            benchmark::DoNotOptimize(s);
        }
    }

    BENCHMARK(BM_Format);

    void BM_Parse(benchmark::State &state) {
        const std::string f = "2014-01-02T03:04:05";
        turbo::CivilSecond c;
        while (state.KeepRunning()) {
            const bool b = turbo::parse_civil_time(f, &c);
            benchmark::DoNotOptimize(b);
        }
    }

    BENCHMARK(BM_Parse);

    void BM_RoundTripFormatParse(benchmark::State &state) {
        const turbo::CivilSecond c(2014, 1, 2, 3, 4, 5);
        turbo::CivilSecond out;
        while (state.KeepRunning()) {
            const bool b = turbo::parse_civil_time(turbo::format_civil_time(c), &out);
            benchmark::DoNotOptimize(b);
        }
    }

    BENCHMARK(BM_RoundTripFormatParse);

    template<typename T>
    void BM_CivilTimeTurboHash(benchmark::State &state) {
        const int kSize = 100000;
        std::vector<T> civil_times(kSize);
        std::iota(civil_times.begin(), civil_times.end(), T(2018));

        turbo::Hash<T> turbo_hasher;
        while (state.KeepRunningBatch(kSize)) {
            for (const T civil_time: civil_times) {
                benchmark::DoNotOptimize(turbo_hasher(civil_time));
            }
        }
    }

    void BM_CivilYearTurboHash(benchmark::State &state) {
        BM_CivilTimeTurboHash<turbo::CivilYear>(state);
    }

    void BM_CivilMonthTurboHash(benchmark::State &state) {
        BM_CivilTimeTurboHash<turbo::CivilMonth>(state);
    }

    void BM_CivilDayTurboHash(benchmark::State &state) {
        BM_CivilTimeTurboHash<turbo::CivilDay>(state);
    }

    void BM_CivilHourTurboHash(benchmark::State &state) {
        BM_CivilTimeTurboHash<turbo::CivilHour>(state);
    }

    void BM_CivilMinuteTurboHash(benchmark::State &state) {
        BM_CivilTimeTurboHash<turbo::CivilMinute>(state);
    }

    void BM_CivilSecondTurboHash(benchmark::State &state) {
        BM_CivilTimeTurboHash<turbo::CivilSecond>(state);
    }

    BENCHMARK(BM_CivilYearTurboHash);
    BENCHMARK(BM_CivilMonthTurboHash);
    BENCHMARK(BM_CivilDayTurboHash);
    BENCHMARK(BM_CivilHourTurboHash);
    BENCHMARK(BM_CivilMinuteTurboHash);
    BENCHMARK(BM_CivilSecondTurboHash);

}  // namespace
