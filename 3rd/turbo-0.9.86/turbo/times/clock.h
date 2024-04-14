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

#ifndef TURBO_TIME_CLOCK_H_
#define TURBO_TIME_CLOCK_H_

#include "turbo/platform/port.h"
#include "turbo/times/time.h"

namespace turbo {

    /**
     * @ingroup turbo_times_clock
     * @brief Returns the current time, expressed as a count of nanoseconds since the Unix Epoch.
     *        Prefer `turbo::time_now()` instead for all but the most performance-sensitive cases
     *        (i.e. when you are calling this function hundreds of thousands of times per second).
     *        @see `turbo::time_now()`
     *        @see `turbo::get_current_time_nanos()`
     * @return
     */
    int64_t get_current_time_nanos();

    inline Time time_now() {
        return Time::time_now();
    }

    inline int64_t get_current_time_micros() {
        return get_current_time_nanos() / 1000;
    }

    /**
     * @ingroup turbo_times_clock
     * @brief Returns the current time, expressed as a count of milliseconds since the Unix Epoch.
     *        Prefer `turbo::time_now()` instead for all but the most performance-sensitive cases
     *        (i.e. when you are calling this function hundreds of thousands of times per second).
     *        @see `turbo::time_now()`
     *        @see `turbo::get_current_time_nanos()`
     * @return
     */
    inline int64_t get_current_time_millis() {
        return get_current_time_nanos() / 1000000;
    }

    /**
     * @ingroup turbo_times_clock
     * @brief Returns the current time, expressed as a count of seconds since the Unix Epoch.
     *        Prefer `turbo::time_now()` instead for all but the most performance-sensitive cases
     *        (i.e. when you are calling this function hundreds of thousands of times per second).
     *        @see `turbo::time_now()`
     *        @see `turbo::get_current_time_nanos()`
     * @return
     */
    inline int64_t get_current_time_seconds() {
        return get_current_time_nanos() / 1000000000;
    }

    /**
     * @ingroup turbo_times_clock
     * @brief Sleeps for the specified duration, expressed as an `turbo::Duration`.
     * @note Signal interruptions will not reduce the sleep duration.
     *       Returns immediately when passed a nonpositive duration.
     *       @see `turbo::Duration`
     * @param duration The duration to sleep for.
     */
    void sleep_for(turbo::Duration duration);

    /**
     * @ingroup turbo_times_clock
     * @brief Sleeps until the specified time, expressed as an `turbo::Time`.
     * @note Signal interruptions will not reduce the sleep duration.
     *       Returns immediately when passed a time in the past.
     *       @see `turbo::Time`
     * @param time The time to sleep until.
     */
    inline void sleep_until(turbo::Time time) {
        sleep_for(time - Time::time_now());
    }

    /**
     * @ingroup turbo_times_clock
     * @brief Returns the current time, expressed as an `turbo::Time` absolute time value.
     * @return
     */
    inline turbo::Time from_now(timespec ts) {
        return Time::time_now() + Duration::from_timespec(ts);
    }

    inline turbo::Time from_now(Duration d) {
        return Time::time_now() + d;
    }

    inline turbo::Time from_now(timeval tv) {
        return Time::time_now() + Duration::from_timeval(tv);
    }

    inline turbo::Time seconds_from_now(int64_t secs) {
        return Time::time_now() + turbo::Duration::seconds(secs);
    }

    inline turbo::Time milliseconds_from_now(int64_t ms) {
        return Time::time_now() + turbo::Duration::milliseconds(ms);
    }

    inline turbo::Time microseconds_from_now(int64_t us) {
        return Time::time_now() + turbo::Duration::microseconds(us);
    }

    inline turbo::Time nanoseconds_from_now(int64_t ns) {
        return Time::time_now() + turbo::Duration::nanoseconds(ns);
    }

    inline turbo::Time double_seconds_from_now(double secs) {
        return Time::time_now() + turbo::Duration::seconds(secs);
    }

    inline turbo::Time double_milliseconds_from_now(double ms) {
        return Time::time_now() + turbo::Duration::milliseconds(ms);
    }

    inline turbo::Time double_microseconds_from_now(double us) {
        return Time::time_now() + turbo::Duration::microseconds(us);
    }

    inline turbo::Time double_nanoseconds_from_now(double ns) {
        return Time::time_now() + turbo::Duration::nanoseconds(ns);
    }


}  // namespace turbo

// -----------------------------------------------------------------------------
// Implementation Details
// -----------------------------------------------------------------------------

// In some build configurations we pass --detect-odr-violations to the
// gold linker.  This causes it to flag weak symbol overrides as ODR
// violations.  Because ODR only applies to C++ and not C,
// --detect-odr-violations ignores symbols not mangled with C++ names.
// By changing our extension points to be extern "C", we dodge this
// check.
extern "C" {
void turbo_internal_sleep_for(turbo::Duration duration);
}  // extern "C"

inline void turbo::sleep_for(turbo::Duration duration) {
    turbo_internal_sleep_for(duration);
}

#endif  // TURBO_TIME_CLOCK_H_
