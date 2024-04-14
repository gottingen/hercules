// Copyright 2023 The titan-search Authors.
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


#ifndef TURBO_LOG_LOGGING_H_
#define TURBO_LOG_LOGGING_H_

#include "turbo/log/tlog.h"
#include "turbo/platform/port.h"
#include "turbo/log/condition.h"


#ifndef TLOG_CRASH_IS_ON
#if TURBO_OPTION_HARDENED == 1
#define TLOG_CRASH_IS_ON() 1
#else
#define TLOG_CRASH_IS_ON() 0
#endif
#endif  // TLOG_CRASH_IS_ON

#ifndef TLOG_DEBUG_IS_ON
#if TURBO_OPTION_DEBUG == 1
#define TLOG_DEBUG_IS_ON() 1
#else
#define TLOG_DEBUG_IS_ON() 0
#endif
#endif  // TLOG_DEBUG_IS_ON

#ifndef TLOG_CRASH
#if TLOG_CRASH_IS_ON()
#define TLOG_CRASH()  std::abort()
#else
#define TLOG_CRASH() (void)0
#endif
#endif  // TLOG_CRASH

//
// enable/disable log calls at compile time according to global level.
//
// define TLOG_ACTIVE_LEVEL to one of those (before including spdlog.h):
// TLOG_LEVEL_TRACE,
// TLOG_LEVEL_DEBUG,
// TLOG_LEVEL_INFO,
// TLOG_LEVEL_WARN,
// TLOG_LEVEL_ERROR,
// TLOG_LEVEL_CRITICAL,
// TLOG_LEVEL_OFF
//

#if TLOG_SOURCE_LOC
#    define TLOG_LOGGER_CALL(logger, level, ...)                                                                                         \
        (logger)->log(turbo::tlog::source_loc{__FILE__, __LINE__, TLOG_FUNCTION}, level, __VA_ARGS__)
#    define TLOG_LOGGER_CALL_IF(logger, level, cond, ...) \
        (cond) ? (logger)->log(turbo::tlog::source_loc{__FILE__, __LINE__, TLOG_FUNCTION}, level, __VA_ARGS__) : (void)0
#    define TLOG_LOGGER_CALL_IF_EVERY_N(logger, level, cond, N, ...) \
        static ::turbo::tlog::details::LogEveryNState TURBO_CONCAT(everyn_, __LINE__);                                                             \
        (TURBO_CONCAT(everyn_, __LINE__).ShouldLog((N)) && cond) ? (logger)->log(turbo::tlog::source_loc{__FILE__, __LINE__, TLOG_FUNCTION}, level, __VA_ARGS__) : (void)0
#    define TLOG_LOGGER_CALL_IF_FIRST_N(logger, level, cond, N, ...) \
        static ::turbo::tlog::details::LogFirstNState TURBO_CONCAT(firstn_, __LINE__);                                                             \
        (TURBO_CONCAT(firstn_, __LINE__).ShouldLog((N)) && cond) ? (logger)->log(turbo::tlog::source_loc{__FILE__, __LINE__, TLOG_FUNCTION}, level, __VA_ARGS__) : (void)0
#    define TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, level, cond, N, ...) \
        static ::turbo::tlog::details::LogEveryNSecState TURBO_CONCAT(everynsec_, __LINE__);                                                             \
        (TURBO_CONCAT(everynsec_, __LINE__).ShouldLog((N)) && cond) ? (logger)->log(turbo::tlog::source_loc{__FILE__, __LINE__, TLOG_FUNCTION}, level, __VA_ARGS__) : (void)0
#else
#    define TLOG_LOGGER_CALL(logger, level, ...) (logger)->log(turbo::tlog::source_loc{}, level, __VA_ARGS__)
#    define TLOG_LOGGER_CALL_IF(logger, level, cond, ...) (cond) ? (logger)->log(turbo::tlog::source_loc{}, level, __VA_ARGS__) : (void)0
#    define TLOG_LOGGER_CALL_IF_EVERY_N(logger, level, cond, N, ...) \
        static ::turbo::tlog::details::LogEveryNState TURBO_CONCAT(everyn_, __LINE__);                                                             \
        (TURBO_CONCAT(everyn_, __LINE__).ShouldLog((N)) && cond) ? (logger)->log(turbo::tlog::source_loc{}, level, __VA_ARGS__) : (void)0
#    define TLOG_LOGGER_CALL_IF_FIRST_N(logger, level, cond, N, ...) \
        static ::turbo::tlog::details::LogFirstNState TURBO_CONCAT(firstn_, __LINE__);                                                             \
        (TURBO_CONCAT(everyn_, __LINE__).ShouldLog((N)) && cond) ? (logger)->log(turbo::tlog::source_loc{}, level, __VA_ARGS__) : (void)0
#    define TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, level, cond, N, ...) \
        static ::turbo::tlog::details::LogEveryNSecState TURBO_CONCAT(everynsec_, __LINE__);                                                             \
        (TURBO_CONCAT(everynsec_, __LINE__).ShouldLog((N)) && cond) ? (logger)->log(turbo::tlog::source_loc{}, level, __VA_ARGS__) : (void)0
#endif

#if TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_TRACE
#    define TLOG_LOGGER_TRACE(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::trace, __VA_ARGS__)
#    define TLOG_LOGGER_TRACE_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::trace, cond, __VA_ARGS__)
#    define TLOG_LOGGER_TRACE_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::trace, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::trace, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::trace, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::trace, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::trace, cond, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::trace, true, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::trace, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::trace, cond, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::trace, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_TRACE_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::trace, true, 1, ##__VA_ARGS__)
#    define TLOG_TRACE(...) TLOG_LOGGER_TRACE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_TRACE_IF(cond, ...) TLOG_LOGGER_TRACE_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_TRACE_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_TRACE_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_TRACE_EVERY_N(N, ...) TLOG_LOGGER_TRACE_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_TRACE_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_TRACE_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_TRACE_FIRST_N(N, ...) TLOG_LOGGER_TRACE_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_TRACE_IF_ONCE(cond, ...) TLOG_LOGGER_TRACE_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_TRACE_ONCE(...) TLOG_LOGGER_TRACE_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_TRACE_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_TRACE_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_TRACE_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_TRACE_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TLOG_TRACE_EVERY_N_SEC(N, ...) TLOG_LOGGER_TRACE_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_TRACE_EVERY_SEC(...) TLOG_LOGGER_TRACE_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TLOG_LOGGER_TRACE(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_IF(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_IF_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_IF_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_IF_EVERY_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_TRACE_EVERY_SEC(logger, ...) (void)0
#    define TLOG_TRACE(...) (void)0
#    define TLOG_TRACE_IF(...) (void)0
#    define TLOG_TRACE_IF_EVERY_N(...) (void)0
#    define TLOG_TRACE_EVERY_N(...) (void)0
#    define TLOG_TRACE_IF_FIRST_N(...) (void)0
#    define TLOG_TRACE_FIRST_N(...) (void)0
#    define TLOG_TRACE_IF_ONCE(...) (void)0
#    define TLOG_TRACE_ONCE(...) (void)0
#    define TLOG_TRACE_IF_EVERY_N_SEC(...) (void)0
#    define TLOG_TRACE_IF_EVERY_SEC(...) (void)0
#    define TLOG_TRACE_EVERY_N_SEC(...) (void)0
#    define TLOG_TRACE_EVERY_SEC(...) (void)0
#endif


#if (TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_TRACE) && TLOG_DEBUG_IS_ON()
#    define TDLOG_LOGGER_TRACE(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::trace, __VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::trace, cond, __VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::trace, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::trace, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::trace, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::trace, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::trace, cond, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::trace, true, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::trace, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::trace, cond, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::trace, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_TRACE_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::trace, true, 1, ##__VA_ARGS__)
#    define TDLOG_TRACE(...) TLOG_LOGGER_TRACE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_TRACE_IF(cond, ...) TLOG_LOGGER_TRACE_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_TRACE_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_TRACE_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_TRACE_EVERY_N(N, ...) TLOG_LOGGER_TRACE_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_TRACE_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_TRACE_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_TRACE_FIRST_N(N, ...) TLOG_LOGGER_TRACE_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_TRACE_IF_ONCE(cond, ...) TLOG_LOGGER_TRACE_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_TRACE_ONCE(...) TLOG_LOGGER_TRACE_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_TRACE_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_TRACE_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_TRACE_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_TRACE_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TDLOG_TRACE_EVERY_N_SEC(N, ...) TLOG_LOGGER_TRACE_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_TRACE_EVERY_SEC(...) TLOG_LOGGER_TRACE_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TDLOG_LOGGER_TRACE(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_IF(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_IF_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_IF_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_IF_ONCE(logger,...) (void)0
#    define TDLOG_LOGGER_TRACE_ONCE(logger,...) (void)0
#    define TDLOG_LOGGER_TRACE_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_IF_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_TRACE_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_TRACE(...) (void)0
#    define TDLOG_TRACE_IF(...) (void)0
#    define TDLOG_TRACE_IF_EVERY_N(...) (void)0
#    define TDLOG_TRACE_EVERY_N(...) (void)0
#    define TDLOG_TRACE_IF_FIRST_N(...) (void)0
#    define TDLOG_TRACE_FIRST_N(...) (void)0
#    define TDLOG_TRACE_IF_ONCE(...) (void)0
#    define TDLOG_TRACE_ONCE(...) (void)0
#    define TDLOG_TRACE_IF_EVERY_N_SEC(...) (void)0
#    define TDLOG_TRACE_IF_EVERY_SEC(...) (void)0
#    define TDLOG_TRACE_EVERY_N_SEC(...) (void)0
#    define TDLOG_TRACE_EVERY_SEC(...) (void)0
#endif

#if TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_DEBUG
#    define TLOG_LOGGER_DEBUG(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::debug, __VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::debug, cond, __VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::debug, cond, N, __VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::debug, true, N, __VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::debug, cond, N, __VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::debug, true, N, __VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::debug, cond, 1, __VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::debug, true, 1, __VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::debug, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::debug, cond, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::debug, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_DEBUG_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::debug, true, 1, ##__VA_ARGS__)
#    define TLOG_DEBUG(...) TLOG_LOGGER_DEBUG(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_DEBUG_IF(cond, ...) TLOG_LOGGER_DEBUG_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_DEBUG_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_DEBUG_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_DEBUG_EVERY_N(cond, N, ...) TLOG_LOGGER_DEBUG_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_DEBUG_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_DEBUG_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_DEBUG_FIRST_N(N, ...) TLOG_LOGGER_DEBUG_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_DEBUG_IF_ONCE(cond, ...) TLOG_LOGGER_DEBUG_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_DEBUG_ONCE(...) TLOG_LOGGER_DEBUG_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_DEBUG_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_DEBUG_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_DEBUG_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_DEBUG_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TLOG_DEBUG_EVERY_N_SEC(N, ...) TLOG_LOGGER_DEBUG_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_DEBUG_EVERY_SEC( ...) TLOG_LOGGER_DEBUG_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TLOG_LOGGER_DEBUG(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_IF(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_IF_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_IF_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_IF_ONECE(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_ONECE(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_IF_EVERY_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_DEBUG_EVERY_SEC(logger, ...) (void)0
#    define TLOG_DEBUG(...) (void)0
#    define TLOG_DEBUG_IF(...) (void)0
#    define TLOG_DEBUG_IF_EVERY_N(...) (void)0
#    define TLOG_DEBUG_EVERY_N(...) (void)0
#    define TLOG_DEBUG_IF_FIRST_N(...) (void)0
#    define TLOG_DEBUG_FIRST_N(...) (void)0
#    define TLOG_DEBUG_IF_ONCE(...) (void)0
#    define TLOG_DEBUG_ONCE(...) (void)0
#    define TLOG_DEBUG_IF_EVERY_N_SEC(...) (void)0
#    define TLOG_DEBUG_IF_EVERY_SEC(...) (void)0
#    define TLOG_DEBUG_EVERY_N_SEC(...) (void)0
#    define TLOG_DEBUG_EVERY_SEC(...) (void)0
#endif

#if (TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_DEBUG) && TLOG_DEBUG_IS_ON()
#    define TDLOG_LOGGER_DEBUG(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::debug, __VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::debug, cond, __VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::debug, cond, N, __VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::debug, true, N, __VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::debug, cond, N, __VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::debug, true, N, __VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::debug, cond, 1, __VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::debug, true, 1, __VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::debug, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::debug, cond, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::debug, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_DEBUG_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::debug, true, 1, ##__VA_ARGS__)
#    define TDLOG_DEBUG(...) TLOG_LOGGER_DEBUG(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_DEBUG_IF(cond, ...) TLOG_LOGGER_DEBUG_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_DEBUG_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_DEBUG_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_DEBUG_EVERY_N(cond, N, ...) TLOG_LOGGER_DEBUG_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_DEBUG_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_DEBUG_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_DEBUG_FIRST_N(N, ...) TLOG_LOGGER_DEBUG_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_DEBUG_IF_ONCE(cond, ...) TLOG_LOGGER_DEBUG_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_DEBUG_ONCE(...) TLOG_LOGGER_DEBUG_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_DEBUG_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_DEBUG_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_DEBUG_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_DEBUG_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TDLOG_DEBUG_EVERY_N_SEC(N, ...) TLOG_LOGGER_DEBUG_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_DEBUG_EVERY_SEC(...) TLOG_LOGGER_DEBUG_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TDLOG_LOGGER_DEBUG(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_IF(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_IF_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_IF_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_IF_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_IF_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_DEBUG_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_DEBUG(...) (void)0
#    define TDLOG_DEBUG_IF(...) (void)0
#    define TDLOG_DEBUG_IF_EVERY_N(...) (void)0
#    define TDLOG_DEBUG_EVERY_N(...) (void)0
#    define TDLOG_DEBUG_IF_FIRST_N(...) (void)0
#    define TDLOG_DEBUG_FIRST_N(...) (void)0
#    define TDLOG_DEBUG_IF_ONCE(...) (void)0
#    define TDLOG_DEBUG_ONCE(...) (void)0
#    define TDLOG_DEBUG_IF_EVERY_N_SEC(...) (void)0
#    define TDLOG_DEBUG_IF_EVERY_SEC(...) (void)0
#    define TDLOG_DEBUG_EVERY_N_SEC(...) (void)0
#    define TDLOG_DEBUG_EVERY_SEC(...) (void)0
#endif

#if TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_INFO
#    define TLOG_LOGGER_INFO(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::info, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::info, cond, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::info, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::info, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::info, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::info, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::info, cond, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::info, true, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::info, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::info, cond, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::info, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_INFO_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::info, true, 1, ##__VA_ARGS__)
#    define TLOG_INFO(...) TLOG_LOGGER_INFO(turbo::tlog::default_logger_raw(), ##__VA_ARGS__)
#    define TLOG_INFO_IF(cond, ...) TLOG_LOGGER_INFO_IF(turbo::tlog::default_logger_raw(), cond, ##__VA_ARGS__)
#    define TLOG_INFO_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_INFO_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, ##__VA_ARGS__)
#    define TLOG_INFO_EVERY_N(N, ...) TLOG_LOGGER_INFO_EVERY_N(turbo::tlog::default_logger_raw(), N, ##__VA_ARGS__)
#    define TLOG_INFO_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_INFO_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, ##__VA_ARGS__)
#    define TLOG_INFO_FIRST_N(N, ...) TLOG_LOGGER_INFO_FIRST_N(turbo::tlog::default_logger_raw(), N, ##__VA_ARGS__)
#    define TLOG_INFO_IF_ONCE(cond, ...) TLOG_LOGGER_INFO_IF_ONCE(turbo::tlog::default_logger_raw(), cond, ##__VA_ARGS__)
#    define TLOG_INFO_ONCE(...) TLOG_LOGGER_INFO_ONCE(turbo::tlog::default_logger_raw(), ##__VA_ARGS__)
#    define TLOG_INFO_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_INFO_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_INFO_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_INFO_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TLOG_INFO_EVERY_N_SEC(N, ...) TLOG_LOGGER_INFO_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_INFO_EVERY_SEC(...) TLOG_LOGGER_INFO_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TLOG_LOGGER_INFO(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_IF(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_IF_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_IF_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_IF_ONCE(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_ONCE(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_IF_EVERY_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_INFO_EVERY_SEC(logger, ...) (void)0
#    define TLOG_INFO(...) (void)0
#    define TLOG_INFO_IF(...) (void)0
#    define TLOG_INFO_IF_EVERY_N(...) (void)0
#    define TLOG_INFO_EVERY_N(...) (void)0
#    define TLOG_INFO_IF_FIRST_N(...) (void)0
#    define TLOG_INFO_FIRST_N(...) (void)0
#    define TLOG_INFO_IF_ONCE(...) (void)0
#    define TLOG_INFO_ONCE(...) (void)0
#    define TLOG_INFO_IF_EVERY_N_SEC(...) (void)0
#    define TLOG_INFO_IF_EVERY_SEC(...) (void)0
#    define TLOG_INFO_EVERY_N_SEC(...) (void)0
#    define TLOG_INFO_EVERY_SEC(...) (void)0
#endif

#if (TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_INFO) && TLOG_DEBUG_IS_ON()
#    define TDLOG_LOGGER_INFO(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::info, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::info, cond, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::info, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::info, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::info, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::info, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::info, cond, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::info, true, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::info, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::info, cond, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::info, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_INFO_EVERY_N_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::info, true, 1, ##__VA_ARGS__)
#    define TDLOG_INFO(...) TLOG_LOGGER_INFO(turbo::tlog::default_logger_raw(), ##__VA_ARGS__)
#    define TDLOG_INFO_IF(cond, ...) TLOG_LOGGER_INFO_IF(turbo::tlog::default_logger_raw(), cond, ##__VA_ARGS__)
#    define TDLOG_INFO_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_INFO_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, ##__VA_ARGS__)
#    define TDLOG_INFO_EVERY_N(N, ...) TLOG_LOGGER_INFO_EVERY_N(turbo::tlog::default_logger_raw(), N, ##__VA_ARGS__)
#    define TDLOG_INFO_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_INFO_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, ##__VA_ARGS__)
#    define TDLOG_INFO_FIRST_N(N, ...) TLOG_LOGGER_INFO_FIRST_N(turbo::tlog::default_logger_raw(), N, ##__VA_ARGS__)
#    define TDLOG_INFO_IF_ONCE(cond, ...) TLOG_LOGGER_INFO_IF_ONCE(turbo::tlog::default_logger_raw(), cond, ##__VA_ARGS__)
#    define TDLOG_INFO_ONCE(...) TLOG_LOGGER_INFO_ONCE(turbo::tlog::default_logger_raw(), ##__VA_ARGS__)
#    define TDLOG_INFO_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_INFO_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_INFO_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_INFO_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TDLOG_INFO_EVERY_N_SEC(N, ...) TLOG_LOGGER_INFO_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_INFO_EVERY_SEC(...) TLOG_LOGGER_INFO_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TDLOG_LOGGER_INFO(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_IF(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_IF_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_IF_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_IF_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_IF_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_INFO_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_INFO(...) (void)0
#    define TDLOG_INFO_IF(...) (void)0
#    define TDLOG_INFO_IF_EVERY_N(...) (void)0
#    define TDLOG_INFO_EVERY_N(...) (void)0
#    define TDLOG_INFO_IF_FIRST_N(...) (void)0
#    define TDLOG_INFO_FIRST_N(...) (void)0
#    define TDLOG_INFO_IF_ONCE(...) (void)0
#    define TDLOG_INFO_ONCE(...) (void)0
#    define TDLOG_INFO_IF_EVERY_N_SEC(...) (void)0
#    define TDLOG_INFO_IF_EVERY_SEC(...) (void)0
#    define TDLOG_INFO_EVERY_N_SEC(...) (void)0
#    define TDLOG_INFO_EVERY_SEC(...) (void)0
#endif

#if TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_WARN
#    define TLOG_LOGGER_WARN(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::warn, __VA_ARGS__)
#    define TLOG_LOGGER_WARN_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::warn, cond, __VA_ARGS__)
#    define TLOG_LOGGER_WARN_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::warn, cond, N, __VA_ARGS__)
#    define TLOG_LOGGER_WARN_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::warn, true, N, __VA_ARGS__)
#    define TLOG_LOGGER_WARN_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::warn, cond, N, __VA_ARGS__)
#    define TLOG_LOGGER_WARN_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::warn, true, N, __VA_ARGS__)
#    define TLOG_LOGGER_WARN_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::warn, cond, 1, __VA_ARGS__)
#    define TLOG_LOGGER_WARN_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::warn, true, 1, __VA_ARGS__)
#    define TLOG_LOGGER_WARN_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::warn, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_WARN_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::warn, cond, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_WARN_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::warn, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_WARN_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::warn, true, 1, ##__VA_ARGS__)
#    define TLOG_WARN(...) TLOG_LOGGER_WARN(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_WARN_IF(cond, ...) TLOG_LOGGER_WARN_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_WARN_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_WARN_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_WARN_EVERY_N(N, ...) TLOG_LOGGER_WARN_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_WARN_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_WARN_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_WARN_FIRST_N(N, ...) TLOG_LOGGER_WARN_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_WARN_IF_ONCE(cond, ...) TLOG_LOGGER_WARN_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_WARN_ONCE(...) TLOG_LOGGER_WARN_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_WARN_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_WARN_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_WARN_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_WARN_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TLOG_WARN_EVERY_N_SEC(N, ...) TLOG_LOGGER_WARN_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_WARN_EVERY_SEC(...) TLOG_LOGGER_WARN_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TLOG_LOGGER_WARN(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_IF(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_IF_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_IF_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_IF_ONCE(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_ONCE(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_IF_EVERY_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_WARN_EVERY_SEC(logger, ...) (void)0
#    define TLOG_WARN(...) (void)0
#    define TLOG_WARN_IF(...) (void)0
#    define TLOG_WARN_IF_EVERY_N(...) (void)0
#    define TLOG_WARN_EVERY_N(...) (void)0
#    define TLOG_WARN_IF_FIRST_N(...) (void)0
#    define TLOG_WARN_FIRST_N(...) (void)0
#    define TLOG_WARN_IF_ONCE(...) (void)0
#    define TLOG_WARN_ONCE(...) (void)0
#    define TLOG_WARN_IF_EVERY_N_SEC(...) (void)0
#    define TLOG_WARN_IF_EVERY_SEC(...) (void)0
#    define TLOG_WARN_EVERY_N_SEC(...) (void)0
#    define TLOG_WARN_EVERY_SEC(...) (void)0
#endif

#if (TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_WARN) && TLOG_DEBUG_IS_ON()
#    define TDLOG_LOGGER_WARN(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::warn, __VA_ARGS__)
#    define TDLOG_LOGGER_WARN_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::warn, cond, __VA_ARGS__)
#    define TDLOG_LOGGER_WARN_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::warn, cond, N, __VA_ARGS__)
#    define TDLOG_LOGGER_WARN_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::warn, true, N, __VA_ARGS__)
#    define TDLOG_LOGGER_WARN_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::warn, cond, N, __VA_ARGS__)
#    define TDLOG_LOGGER_WARN_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::warn, true, N, __VA_ARGS__)
#    define TDLOG_LOGGER_WARN_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::warn, cond, 1, __VA_ARGS__)
#    define TDLOG_LOGGER_WARN_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::warn, true, 1, __VA_ARGS__)
#    define TDLOG_LOGGER_WARN_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::warn, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_WARN_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::warn, cond, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_WARN_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::warn, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_WARN_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::warn, true, 1, ##__VA_ARGS__)
#    define TDLOG_WARN(...) TLOG_LOGGER_WARN(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_WARN_IF(cond, ...) TLOG_LOGGER_WARN_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_WARN_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_WARN_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_WARN_EVERY_N(N, ...) TLOG_LOGGER_WARN_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_WARN_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_WARN_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_WARN_FIRST_N(N, ...) TLOG_LOGGER_WARN_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_WARN_IF_ONCE(cond, ...) TLOG_LOGGER_WARN_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_WARN_ONCE(...) TLOG_LOGGER_WARN_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_WARN_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_WARN_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_WARN_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_WARN_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TDLOG_WARN_EVERY_N_SEC(N, ...) TLOG_LOGGER_WARN_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_WARN_EVERY_SEC( ...) TLOG_LOGGER_WARN_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TDLOG_LOGGER_WARN(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_IF(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_IF_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_IF_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_IF_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_IF_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_WARN_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_WARN(...) (void)0
#    define TDLOG_WARN_IF(...) (void)0
#    define TDLOG_WARN_IF_EVERY_N(...) (void)0
#    define TDLOG_WARN_EVERY_N(...) (void)0
#    define TDLOG_WARN_IF_FIRST_N(...) (void)0
#    define TDLOG_WARN_FIRST_N(...) (void)0
#    define TDLOG_WARN_IF_ONCE(...) (void)0
#    define TDLOG_WARN_ONCE(...) (void)0
#    define TDLOG_WARN_IF_EVERY_N_SEC(...) (void)0
#    define TDLOG_WARN_IF_EVERY_SEC(...) (void)0
#    define TDLOG_WARN_EVERY_N_SEC(...) (void)0
#    define TDLOG_WARN_EVERY_SEC(...) (void)0
#endif


#if TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_ERROR
#    define TLOG_LOGGER_ERROR(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::err, __VA_ARGS__)
#    define TLOG_LOGGER_ERROR_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::err, cond, __VA_ARGS__)
#    define TLOG_LOGGER_ERROR_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::err, cond, N,__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::err, true, N,__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::err, cond, N,__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::err, true, N,__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::err, cond, 1,__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::err, true, 1,__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::err, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::err, cond, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::err, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_ERROR_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::err, true, 1, ##__VA_ARGS__)
#    define TLOG_ERROR(...) TLOG_LOGGER_ERROR(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_ERROR_IF(cond, ...) TLOG_LOGGER_ERROR_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_ERROR_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_ERROR_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_ERROR_EVERY_N(N, ...) TLOG_LOGGER_ERROR_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_ERROR_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_ERROR_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_ERROR_FIRST_N(N, ...) TLOG_LOGGER_ERROR_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_ERROR_IF_ONCE(cond, ...) TLOG_LOGGER_ERROR_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_ERROR_ONCE(...) TLOG_LOGGER_ERROR_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_ERROR_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_ERROR_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_ERROR_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_ERROR_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TLOG_ERROR_EVERY_N_SEC(N, ...) TLOG_LOGGER_ERROR_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_ERROR_EVERY_SEC(...) TLOG_LOGGER_ERROR_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TLOG_LOGGER_ERROR(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_IF(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_IF_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_IF_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_IF_ONCE(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_ONCE(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_IF_EVERY_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_ERROR_EVERY_SEC(logger, ...) (void)0
#    define TLOG_ERROR(...) (void)0
#    define TLOG_ERROR_IF(...) (void)0
#    define TLOG_ERROR_IF_EVERY_N(...) (void)0
#    define TLOG_ERROR_EVERY_N(...) (void)0
#    define TLOG_ERROR_IF_FIRST_N(...) (void)0
#    define TLOG_ERROR_FIRST_N(...) (void)0
#    define TLOG_ERROR_IF_ONCE(...) (void)0
#    define TLOG_ERROR_ONCE(...) (void)0
#    define TLOG_ERROR_IF_EVERY_N_SEC(...) (void)0
#    define TLOG_ERROR_IF_EVERY_SEC(...) (void)0
#    define TLOG_ERROR_EVERY_N_SEC(...) (void)0
#    define TLOG_ERROR_EVERY_SEC(...) (void)0
#endif

#if (TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_ERROR) && TLOG_DEBUG_IS_ON()
#    define TDLOG_LOGGER_ERROR(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::err, __VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::err, cond, __VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::err, cond, N,__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::err, true, N,__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::err, cond, N,__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::err, true, N,__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::err, cond, 1,__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::err, true, 1,__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::err, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_IF_EVERY_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::err, cond, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::err, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_ERROR_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::err, true, 1, ##__VA_ARGS__)
#    define TDLOG_ERROR(...) TLOG_LOGGER_ERROR(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_ERROR_IF(cond, ...) TLOG_LOGGER_ERROR_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_ERROR_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_ERROR_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_ERROR_EVERY_N(N, ...) TLOG_LOGGER_ERROR_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_ERROR_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_ERROR_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_ERROR_FIRST_N(N, ...) TLOG_LOGGER_ERROR_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_ERROR_IF_ONCE(cond, ...) TLOG_LOGGER_ERROR_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_ERROR_ONCE(...) TLOG_LOGGER_ERROR_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_ERROR_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_ERROR_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_ERROR_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_ERROR_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TDLOG_ERROR_EVERY_N_SEC(N, ...) TLOG_LOGGER_ERROR_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_ERROR_EVERY_SEC(...) TLOG_LOGGER_ERROR_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TDLOG_LOGGER_ERROR(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_IF(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_IF_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_IF_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_IF_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_IF_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_ERROR_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_ERROR(...) (void)0
#    define TDLOG_ERROR_IF(...) (void)0
#    define TDLOG_ERROR_IF_EVERY_N(...) (void)0
#    define TDLOG_ERROR_EVERY_N(...) (void)0
#    define TDLOG_ERROR_IF_FIRST_N(...) (void)0
#    define TDLOG_ERROR_FIRST_N(...) (void)0
#    define TDLOG_ERROR_IF_ONCE(...) (void)0
#    define TDLOG_ERROR_ONCE(...) (void)0
#    define TDLOG_ERROR_IF_EVERY_N_SEC(...) (void)0
#    define TDLOG_ERROR_IF_EVERY_SEC(...) (void)0
#    define TDLOG_ERROR_EVERY_N_SEC(...) (void)0
#    define TDLOG_ERROR_EVERY_SEC(...) (void)0
#endif

#if TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_CRITICAL
#    define TLOG_LOGGER_CRITICAL(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::critical, __VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::critical, cond, __VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::critical, cond, N, __VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::critical, true, N, __VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::critical, cond, N, __VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::critical, true, N, __VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::critical, cond, 1, __VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::critical, true, 1, __VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::critical, cond, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::critical, cond, 1, ##__VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::critical, true, N, ##__VA_ARGS__)
#    define TLOG_LOGGER_CRITICAL_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::critical, true, 1, ##__VA_ARGS__)
#    define TLOG_CRITICAL(...) TLOG_LOGGER_CRITICAL(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_CRITICAL_IF(cond, ...) TLOG_LOGGER_CRITICAL_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_CRITICAL_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_CRITICAL_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_CRITICAL_EVERY_N(N, ...) TLOG_LOGGER_CRITICAL_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_CRITICAL_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_CRITICAL_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_CRITICAL_FIRST_N(N, ...) TLOG_LOGGER_CRITICAL_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_CRITICAL_IF_ONCE(cond, ...) TLOG_LOGGER_CRITICAL_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TLOG_CRITICAL_ONCE(...) TLOG_LOGGER_CRITICAL_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TLOG_CRITICAL_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_CRITICAL_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TLOG_CRITICAL_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_CRITICAL_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TLOG_CRITICAL_EVERY_N_SEC(N, ...) TLOG_LOGGER_CRITICAL_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TLOG_CRITICAL_EVERY_SEC(...) TLOG_LOGGER_CRITICAL_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TLOG_LOGGER_CRITICAL(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_IF(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_IF_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_EVERY_N(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_IF_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_FIRST_N(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_IF_ONCE(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_ONCE(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_IF_EVERY_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_EVERY_N_SEC(logger, ...) (void)0
#    define TLOG_LOGGER_CRITICAL_EVERY_SEC(logger, ...) (void)0
#    define TLOG_CRITICAL(...) (void)0
#    define TLOG_CRITICAL_IF(...) (void)0
#    define TLOG_CRITICAL_IF_EVERY_N(...) (void)0
#    define TLOG_CRITICAL_EVERY_N(...) (void)0
#    define TLOG_CRITICAL_IF_FIRST_N(...) (void)0
#    define TLOG_CRITICAL_FIRST_N(...) (void)0
#    define TLOG_CRITICAL_IF_ONCE(...) (void)0
#    define TLOG_CRITICAL_ONCE(...) (void)0
#    define TLOG_CRITICAL_IF_EVERY_N_SEC(...) (void)0
#    define TLOG_CRITICAL_IF_EVERY_SEC(...) (void)0
#    define TLOG_CRITICAL_EVERY_N_SEC(...) (void)0
#    define TLOG_CRITICAL_EVERY_SEC(...) (void)0
#endif


#if (TLOG_ACTIVE_LEVEL <= TLOG_LEVEL_CRITICAL)&& TLOG_DEBUG_IS_ON()
#    define TDLOG_LOGGER_CRITICAL(logger, ...) TLOG_LOGGER_CALL(logger, turbo::tlog::level::critical, __VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_IF(logger, cond, ...) TLOG_LOGGER_CALL_IF(logger, turbo::tlog::level::critical, cond, __VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_IF_EVERY_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::critical, cond, N, __VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_EVERY_N(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N(logger, turbo::tlog::level::critical, true, N, __VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_IF_FIRST_N(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::critical, cond, N, __VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_FIRST_N(logger, N, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::critical, true, N, __VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_IF_ONCE(logger, cond, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::critical, cond, 1, __VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_ONCE(logger, ...) TLOG_LOGGER_CALL_IF_FIRST_N(logger, turbo::tlog::level::critical, true, 1, __VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_IF_EVERY_N_SEC(logger, cond, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::critical, cond, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_IF_EVERY_SEC(logger, cond, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::critical, cond, 1, ##__VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_EVERY_N_SEC(logger, N, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::critical, true, N, ##__VA_ARGS__)
#    define TDLOG_LOGGER_CRITICAL_EVERY_SEC(logger, ...) TLOG_LOGGER_CALL_IF_EVERY_N_SEC(logger, turbo::tlog::level::critical, true, 1, ##__VA_ARGS__)
#    define TDLOG_CRITICAL(...) TLOG_LOGGER_CRITICAL(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_CRITICAL_IF(cond, ...) TLOG_LOGGER_CRITICAL_IF(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_CRITICAL_IF_EVERY_N(cond, N, ...) TLOG_LOGGER_CRITICAL_IF_EVERY_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_CRITICAL_EVERY_N(N, ...) TLOG_LOGGER_CRITICAL_EVERY_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_CRITICAL_IF_FIRST_N(cond, N, ...) TLOG_LOGGER_CRITICAL_IF_FIRST_N(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_CRITICAL_FIRST_N(N, ...) TLOG_LOGGER_CRITICAL_FIRST_N(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_CRITICAL_IF_ONCE(cond, ...) TLOG_LOGGER_CRITICAL_IF_ONCE(turbo::tlog::default_logger_raw(), cond, __VA_ARGS__)
#    define TDLOG_CRITICAL_ONCE(...) TLOG_LOGGER_CRITICAL_ONCE(turbo::tlog::default_logger_raw(), __VA_ARGS__)
#    define TDLOG_CRITICAL_IF_EVERY_N_SEC(cond, N, ...) TLOG_LOGGER_CRITICAL_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, N, __VA_ARGS__)
#    define TDLOG_CRITICAL_IF_EVERY_SEC(cond, ...) TLOG_LOGGER_CRITICAL_IF_EVERY_N_SEC(turbo::tlog::default_logger_raw(), cond, 1, __VA_ARGS__)
#    define TDLOG_CRITICAL_EVERY_N_SEC(N, ...) TLOG_LOGGER_CRITICAL_EVERY_N_SEC(turbo::tlog::default_logger_raw(), N, __VA_ARGS__)
#    define TDLOG_CRITICAL_EVERY_SEC( ...) TLOG_LOGGER_CRITICAL_EVERY_N_SEC(turbo::tlog::default_logger_raw(), 1, __VA_ARGS__)
#else
#    define TDLOG_LOGGER_CRITICAL(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_IF(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_IF_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_EVERY_N(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_IF_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_FIRST_N(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_IF_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_ONCE(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_IF_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_IF_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_EVERY_N_SEC(logger, ...) (void)0
#    define TDLOG_LOGGER_CRITICAL_EVERY_SEC(logger, ...) (void)0
#    define TDLOG_CRITICAL(...) (void)0
#    define TDLOG_CRITICAL_IF(...) (void)0
#    define TDLOG_CRITICAL_IF_EVERY_N(...) (void)0
#    define TDLOG_CRITICAL_EVERY_N(...) (void)0
#    define TDLOG_CRITICAL_IF_FIRST_N(...) (void)0
#    define TDLOG_CRITICAL_FIRST_N(...) (void)0
#    define TDLOG_CRITICAL_IF_ONCE(...) (void)0
#    define TDLOG_CRITICAL_ONCE(...) (void)0
#    define TDLOG_CRITICAL_IF_EVERY_N_SEC(...) (void)0
#    define TDLOG_CRITICAL_IF_EVERY_SEC(...) (void)0
#    define TDLOG_CRITICAL_EVERY_N_SEC(...) (void)0
#    define TDLOG_CRITICAL_EVERY_SEC(...) (void)0
#endif

namespace turbo::tlog::details {

    template<typename ...Args>
    std::string FormatLog(const std::string_view &prefix, const std::string_view &expr, Args &&...args) {
        std::string result;
        result.append(prefix);
        result.append(expr);
        if constexpr (sizeof...(args) != 0) {
            result.append(" ");
            turbo::format_append(&result, std::forward<Args>(args)...);
        }
        return result;
    }

    inline std::string MakeCheckString(std::string_view name, std::string_view v1, std::string_view v2, std::string_view op) {
        return turbo::format("Check {} failed: {} {} {}", name, v1, op, v2);
    }

}  // namespace turbo::tlog::details

#define TLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(name, op, val1, val2, ...)    \
  {                                                                       \
    auto&& tlog_anonymous_x = (val1);                                       \
    auto&& tlog_anonymous_y = (val2);                                       \
    if (TURBO_UNLIKELY(!(tlog_anonymous_x op tlog_anonymous_y))) {          \
        TLOG_LOGGER_CRITICAL(turbo::tlog::default_logger_raw(), ::turbo::tlog::details::FormatLog("", ::turbo::tlog::details::MakeCheckString(#name, #val1, #val2, #op), ##__VA_ARGS__)); \
        TLOG_CRASH();                                                                        \
    }                                                                        \
  }

#define TDLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(name, op, val1, val2, ...)    \
  if constexpr (TLOG_DEBUG_IS_ON()) {                                                                       \
    auto&& tlog_anonymous_x = (val1);                                       \
    auto&& tlog_anonymous_y = (val2);                                       \
    if (TURBO_UNLIKELY(!(tlog_anonymous_x op tlog_anonymous_y))) {          \
        TLOG_LOGGER_CRITICAL(turbo::tlog::default_logger_raw(), ::turbo::tlog::details::FormatLog("", ::turbo::tlog::details::MakeCheckString(#name, #val1, #val2, #op), ##__VA_ARGS__)); \
        TLOG_CRASH();                                                                        \
    }                                                                        \
  }

#define TLOG_CHECK(expr, ...) \
           if(TURBO_UNLIKELY(!(expr))) {        \
              TLOG_LOGGER_CRITICAL(turbo::tlog::default_logger_raw(), ::turbo::tlog::details::FormatLog("Check failed: ", #expr, ##__VA_ARGS__)); \
              TLOG_CRASH();                \
           }                    \

#define TDLOG_CHECK(expr, ...) \
       if constexpr (TLOG_DEBUG_IS_ON()) {                   \
           if(TURBO_UNLIKELY(!(expr))) {        \
                TLOG_LOGGER_CRITICAL(turbo::tlog::default_logger_raw(), ::turbo::tlog::details::FormatLog("Check failed: ", #expr, ##__VA_ARGS__)); \
                TLOG_CRASH();                \
           }                    \
       }

#define TLOG_CHECK_EQ(val1, val2, ...) \
    TLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckEQ, ==, val1, val2, ##__VA_ARGS__)


#define TDLOG_CHECK_EQ(val1, val2, ...) \
    TDLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckEQ, ==, val1, val2, ##__VA_ARGS__)

#define TLOG_CHECK_NE(val1, val2, ...) \
    TLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckNE, !=, val1, val2, ##__VA_ARGS__)

#define TLOG_CHECK_LE(val1, val2, ...) \
    TLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckLE, <=, val1, val2, ##__VA_ARGS__)

#define TLOG_CHECK_LT(val1, val2, ...) \
    TLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckLE, <, val1, val2, ##__VA_ARGS__)

#define TLOG_CHECK_GE(val1, val2, ...) \
    TLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckGE, >=, val1, val2, ##__VA_ARGS__)

#define TLOG_CHECK_GT(val1, val2, ...) \
    TLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckGT, >, val1, val2, ##__VA_ARGS__)

#define TDLOG_CHECK_EQ(val1, val2, ...) \
    TDLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckEQ, ==, val1, val2, ##__VA_ARGS__)

#define TDLOG_CHECK_NE(val1, val2, ...) \
    TDLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckNE, !=, val1, val2, ##__VA_ARGS__)

#define TDLOG_CHECK_LE(val1, val2, ...) \
    TDLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckLE, <=, val1, val2, ##__VA_ARGS__)

#define TDLOG_CHECK_LT(val1, val2, ...) \
    TDLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckLE, <, val1, val2, ##__VA_ARGS__)

#define TDLOG_CHECK_GE(val1, val2, ...) \
    TDLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckGE, >=, val1, val2, ##__VA_ARGS__)

#define TDLOG_CHECK_GT(val1, val2, ...) \
    TDLOG_INTERNAL_DETAIL_LOGGING_CHECK_OP(CheckGT, >, val1, val2, ##__VA_ARGS__)


#endif  // TURBO_LOG_LOGGING_H_
