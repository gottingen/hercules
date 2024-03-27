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

#include <chrono>
#include <collie/strings/fmt/format.h>

// Stopwatch support for clog  (using std::chrono::steady_clock).
// Displays elapsed seconds since construction as double.
//
// Usage:
//
// clog::stopwatch sw;
// ...
// clog::debug("Elapsed: {} seconds", sw);    =>  "Elapsed 0.005116733 seconds"
// clog::info("Elapsed: {:.6} seconds", sw);  =>  "Elapsed 0.005163 seconds"
//
//
// If other units are needed (e.g. millis instead of double), include <collie/strings/fmt/chrono.h> and use
// "duration_cast<..>(sw.elapsed())":
//
// #include <collie/log/fmt/chrono.h>
//..
// using std::chrono::duration_cast;
// using std::chrono::milliseconds;
// clog::info("Elapsed {}", duration_cast<milliseconds>(sw.elapsed())); => "Elapsed 5ms"

namespace clog {
    class stopwatch {
        using clock = std::chrono::steady_clock;
        std::chrono::time_point<clock> start_tp_;

    public:
        stopwatch()
                : start_tp_{clock::now()} {}

        std::chrono::duration<double> elapsed() const {
            return std::chrono::duration<double>(clock::now() - start_tp_);
        }

        void reset() { start_tp_ = clock::now(); }
    };
}  // namespace clog

// Support for fmt formatting  (e.g. "{:012.9}" or just "{}")
namespace fmt {

    template<>
    struct formatter<clog::stopwatch> : formatter<double> {
        template<typename FormatContext>
        auto format(const clog::stopwatch &sw, FormatContext &ctx) const -> decltype(ctx.out()) {
            return formatter<double>::format(sw.elapsed().count(), ctx);
        }
    };
}  // namespace std
