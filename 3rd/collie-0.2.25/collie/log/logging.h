// Copyright 2023 The Elastic-AI Authors.
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

#include <collie/log/clog.h>
#include <collie/strings/format.h>
#include <collie/log/details/condition.h>
#include <collie/base/macros.h>
#include <system_error>

namespace clog::details {

#ifdef NDEBUG
    static constexpr bool debug_mode = false;
#else
    static constexpr bool debug_mode = true;
#endif

    template<bool PLOG>
    class LogStream {
    public:
        LogStream(logger *logger, source_loc loc, level::level_enum level)
                : logger_(logger), loc_(loc), level_(level) {
            if (PLOG) {
                plog_string();
            }
        }

        ~LogStream() {
            logger_->log(loc_, level_, details::to_string_view(stream_));
        }

        template<typename T>
        LogStream &operator<<(const T &data) {
            collie::format_to(fmt::appender(stream_), "{}", data);
            return *this;
        }

    private:
        void plog_string() {
            auto err = errno;
            auto ec = std::make_error_code(static_cast<std::errc>(err));
            collie::format_to(collie::appender(stream_), "[error_code: {}, {}] ", ec.value(), ec.message());
        }

    protected:
        memory_buf_t stream_;
        logger *logger_;
        source_loc loc_;
        level::level_enum level_;
    };

    class NullStream {
    public:
        constexpr NullStream(logger *logger, source_loc loc, level::level_enum level) {}

        ~NullStream() = default;

        template<typename T>
        constexpr NullStream &operator<<(const T &) {
            return *this;
        }
    };


    struct LogMessageVoidify {
        // This has to be an operator with a precedence lower than << but
        // higher than ?:
        template<bool P>
        void operator&(LogStream<P> &) noexcept {}
    };

    inline std::string
    make_check_string(std::string_view name, std::string_view v1, std::string_view v2, std::string_view op) {
        return collie::format("Check {} failed: {} {} {}", name, v1, op, v2);
    }

    template<typename T>
    inline void make_check_op_value_string(memory_buf_t *stream, const T &v) {
        collie::format_to(collie::appender(*stream), "{}", v);
    }

    template<>
    inline void make_check_op_value_string(memory_buf_t *stream, const char &v) {
        if (v >= 32 && v <= 126) {
            collie::format_to(collie::appender(*stream), "{}", v);
        } else {
            collie::format_to(collie::appender(*stream), "char value {}", static_cast<short>(v));
        }
    }

    template<>
    inline void make_check_op_value_string(memory_buf_t *stream, const signed char &v) {
        if (v >= 32 && v <= 126) {
            collie::format_to(collie::appender(*stream), "{}", v);
        } else {
            collie::format_to(collie::appender(*stream), "char value {}", static_cast<short>(v));
        }
    }

    template<>
    inline void make_check_op_value_string(memory_buf_t *stream, const unsigned char &v) {
        if (v >= 32 && v <= 126) {
            collie::format_to(collie::appender(*stream), "{}", v);
        } else {
            collie::format_to(collie::appender(*stream), "char value {}", static_cast<short>(v));
        }
    }

    template<>
    inline void make_check_op_value_string(memory_buf_t *stream, const std::nullptr_t &) {
        collie::format_to(collie::appender(*stream), "nullptr");
    }


    class CheckOpMessageBuilder {
    public:
        // Inserts "exprtext" and " (" to the stream.
        explicit CheckOpMessageBuilder(const char *exprtext) {
            collie::format_to(collie::appender(stream_), "{} (", exprtext);
        }

        // For inserting the first variable.
        template<typename T>
        void for_var1(const T &v) {
            make_check_op_value_string(&stream_, v);
        }


        // For inserting the second variable (adds an intermediate " vs. ").
        template<typename U>
        void for_var2(const U &v) {
            collie::format_to(collie::appender(stream_), " vs. ");
            make_check_op_value_string(&stream_, v);
            collie::format_to(collie::appender(stream_), ")");

        }

        std::string to_string() const {
            return std::string(stream_.data(), stream_.size());
        }

    private:
        memory_buf_t stream_;
    };

    template<typename T>
    inline void MakeCheckOpValueString(std::ostream *os, const T &v) {
        (*os) << v;
    }

    template<typename T1, typename T2>
    std::string MakeCheckOpString(const T1 &v1, const T2 &v2, const char *exprtext) {
        CheckOpMessageBuilder comb(exprtext);
        comb.for_var1(v1);
        comb.for_var2(v2);
        return comb.to_string();
    }

    template<class T>
    inline const T &get_referenceable_value(const T &t) {
        return t;
    }

    inline char get_referenceable_value(char t) { return t; }

    inline unsigned char get_referenceable_value(unsigned char t) { return t; }

    inline signed char get_referenceable_value(signed char t) { return t; }

    inline short get_referenceable_value(short t) { return t; }

    inline unsigned short get_referenceable_value(unsigned short t) { return t; }

    inline int get_referenceable_value(int t) { return t; }

    inline unsigned int get_referenceable_value(unsigned int t) { return t; }

    inline long get_referenceable_value(long t) { return t; }

    inline unsigned long get_referenceable_value(unsigned long t) { return t; }

    inline long long get_referenceable_value(long long t) { return t; }

    inline unsigned long long get_referenceable_value(unsigned long long t) {
        return t;
    }

}  // namespace clog:details

#define IS_VLOG_ON_LOGGER(verboselevel, logger) ((logger)->vlog_level() > (verboselevel))
#define IS_VLOG_ON(verboselevel) IS_VLOG_ON_LOGGER(verboselevel, clog::default_logger_raw())

#if defined(NDEBUG) && !defined(CLOG_CHECK_ALWAYS_ON)
#  define CLOG_DCHECK_IS_ON() 0
#else
#  define CLOG_DCHECK_IS_ON() 1
#endif

#ifndef CLOG_NO_SOURCE_LOC
#define COLLIE_LOG_LOGGER_STREAM(logger, level, pl) \
        !logger->should_log(level) ? (void)0 : clog::details::LogMessageVoidify()&clog::details::LogStream<pl>((logger), clog::source_loc{__FILE__, __LINE__, CLOG_FUNCTION}, level)
#else
#define COLLIE_LOG_LOGGER_STREAM(logger, level) \
        clog::LogStream((logger), clog::source_loc{}, level)
#endif

#define COLLIE_LOG_LOGGER_NULL_STREAM() \
        clog::details::NullStream((nullptr), clog::source_loc{}, clog::level::trace)

#define COLLIE_LOG_TRACE(logger, pl) COLLIE_LOG_LOGGER_STREAM(logger, clog::level::trace, pl)
#define COLLIE_LOG_DEBUG(logger, pl) COLLIE_LOG_LOGGER_STREAM(logger, clog::level::debug, pl)
#define COLLIE_LOG_INFO(logger, pl) COLLIE_LOG_LOGGER_STREAM(logger, clog::level::info, pl)
#define COLLIE_LOG_WARN(logger, pl) COLLIE_LOG_LOGGER_STREAM(logger, clog::level::warn, pl)
#define COLLIE_LOG_ERROR(logger, pl) COLLIE_LOG_LOGGER_STREAM(logger, clog::level::error, pl)
#define COLLIE_LOG_FATAL(logger, pl) COLLIE_LOG_LOGGER_STREAM(logger, clog::level::fatal, pl)


#define COLLIE_LOGGER_IF_CALL_EVERY_N(SEVERITY, N, condition, logger, pl) \
        static ::clog::details::LogEveryNState COLLIE_CONCAT(everyn_, __LINE__); \
        if(COLLIE_CONCAT(everyn_, __LINE__).should_log((N)) && logger->should_log(static_cast<clog::level::level_enum>(CLOG_LEVEL_##SEVERITY)) && (condition)) \
            clog::details::LogMessageVoidify()&clog::details::LogStream<pl>((logger), clog::source_loc{__FILE__, __LINE__, CLOG_FUNCTION}, static_cast<clog::level::level_enum>(CLOG_LEVEL_##SEVERITY))

#define COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, N, condition, logger, pl) \
        static ::clog::details::LogFirstNState COLLIE_CONCAT(firstn_, __LINE__); \
        if(COLLIE_CONCAT(firstn_, __LINE__).should_log((N)) && logger->should_log(static_cast<clog::level::level_enum>(CLOG_LEVEL_##SEVERITY)) && (condition)) \
            clog::details::LogMessageVoidify()&clog::details::LogStream<pl>((logger), clog::source_loc{__FILE__, __LINE__, CLOG_FUNCTION}, static_cast<clog::level::level_enum>(CLOG_LEVEL_##SEVERITY))

#define COLLIE_LOGGER_IF_CALL_DURATION(SEVERITY, seconds, condition, logger, pl) \
        constexpr size_t COLLIE_CONCAT(LOG_TIME_PERIOD, __LINE__) =                         \
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(seconds)).count(); \
        static ::clog::details::LogEveryNDurationState<COLLIE_CONCAT(LOG_TIME_PERIOD, __LINE__)> COLLIE_CONCAT(every_t_, __LINE__); \
        if(COLLIE_CONCAT(every_t_, __LINE__).should_log() && logger->should_log(static_cast<clog::level::level_enum>(CLOG_LEVEL_##SEVERITY)) && (condition)) \
            clog::details::LogMessageVoidify()&clog::details::LogStream<pl>((logger), clog::source_loc{__FILE__, __LINE__, CLOG_FUNCTION}, static_cast<clog::level::level_enum>(CLOG_LEVEL_##SEVERITY))

#define COLLIE_LOG_IMPL(SEVERITY, logger, pl) COLLIE_LOG_##SEVERITY(logger, pl)

#define LOG_LOGGER(SEVERITY, logger) COLLIE_LOG_IMPL(SEVERITY, logger, false)
#define LOG_LOGGER_IF(SEVERITY, condition, logger) if((condition))  COLLIE_LOG_IMPL(SEVERITY, logger, false)
#define LOG_LOGGER_EVERY_N(SEVERITY, N, logger) COLLIE_LOGGER_IF_CALL_EVERY_N(SEVERITY, (N), (true), logger, false)
#define LOG_LOGGER_IF_EVERY_N(SEVERITY, N, condition, logger) COLLIE_LOGGER_IF_CALL_EVERY_N(SEVERITY, (N), (condition), logger, false)
#define LOG_LOGGER_FIRST_N(SEVERITY, N, logger) COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, (N), true, logger, false)
#define LOG_LOGGER_IF_FIRST_N(SEVERITY, N, condition, logger) COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, (N), (condition), logger, false)
#define LOG_LOGGER_EVERY_T(SEVERITY, seconds, logger) COLLIE_LOGGER_IF_CALL_DURATION(SEVERITY, (seconds), (true), logger, false)
#define LOG_LOGGER_IF_EVERY_T(SEVERITY, seconds, condition, logger) COLLIE_LOGGER_IF_CALL_DURATION(SEVERITY, (seconds), (condition), logger, false)
#define LOG_LOGGER_ONCE(SEVERITY, logger) COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, 1, true, logger, false)
#define LOG_LOGGER_IF_ONCE(SEVERITY, condition, logger) COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, 1, (condition), logger, false)


#define PLOG_LOGGER(SEVERITY, logger) COLLIE_LOG_IMPL(SEVERITY, logger, true)
#define PLOG_LOGGER_IF(SEVERITY, condition, logger) if((condition)) COLLIE_LOG_IMPL(SEVERITY, logger, true)
#define PLOG_LOGGER_EVERY_N(SEVERITY, N, logger) COLLIE_LOGGER_IF_CALL_EVERY_N(SEVERITY, (N), (true), logger, true)
#define PLOG_LOGGER_IF_EVERY_N(SEVERITY, N, condition, logger) COLLIE_LOGGER_IF_CALL_EVERY_N(SEVERITY, (N), (condition), logger, true)
#define PLOG_LOGGER_FIRST_N(SEVERITY, N, logger) COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, (N), true, logger, true)
#define PLOG_LOGGER_IF_FIRST_N(SEVERITY, N, condition, logger) COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, (N), (condition), logger, true)
#define PLOG_LOGGER_EVERY_T(SEVERITY, seconds, logger) COLLIE_LOGGER_IF_CALL_DURATION(SEVERITY, (seconds), (true), logger, true)
#define PLOG_LOGGER_IF_EVERY_T(SEVERITY, seconds, condition, logger) COLLIE_LOGGER_IF_CALL_DURATION(SEVERITY, (seconds), (condition), logger, true)
#define PLOG_LOGGER_ONCE(SEVERITY, logger) COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, 1, true, logger, true)
#define PLOG_LOGGER_IF_ONCE(SEVERITY, condition, logger) COLLIE_LOGGER_IF_CALL_FIRST_N(SEVERITY, 1, (condition), logger, true)

#define VLOG_LOGGER(verboselevel, logger)  LOG_LOGGER_IF(INFO, IS_VLOG_ON_LOGGER(verboselevel, logger), logger)
#define VLOG_IF_LOGGER(verboselevel, condition, logger) LOG_LOGGER_IF(INFO, IS_VLOG_ON_LOGGER(verboselevel, logger) && (condition), logger)
#define VLOG_EVERY_N_LOGGER(verboselevel, N, logger) LOG_LOGGER_IF_EVERY_N(INFO, N, IS_VLOG_ON_LOGGER(verboselevel, logger), logger)
#define VLOG_IF_EVERY_N_LOGGER(verboselevel, N, condition, logger) LOG_LOGGER_IF_EVERY_N(INFO, N, IS_VLOG_ON_LOGGER(verboselevel, logger) && (condition), logger)
#define VLOG_FIRST_N_LOGGER(verboselevel, N, logger) LOG_LOGGER_IF_FIRST_N(INFO, N, IS_VLOG_ON_LOGGER(verboselevel, logger), logger)
#define VLOG_IF_FIRST_N_LOGGER(verboselevel, N, condition, logger) LOG_LOGGER_IF_FIRST_N(INFO, N, IS_VLOG_ON_LOGGER(verboselevel, logger) && (condition), logger)
#define VLOG_EVERY_T_LOGGER(verboselevel, seconds, logger) LOG_LOGGER_IF_EVERY_T(INFO, seconds, IS_VLOG_ON_LOGGER(verboselevel, logger), logger)
#define VLOG_IF_EVERY_T_LOGGER(verboselevel, seconds, condition, logger) LOG_LOGGER_IF_EVERY_T(INFO, seconds, IS_VLOG_ON_LOGGER(verboselevel, logger) && (condition), logger)
#define VLOG_ONCE_LOGGER(verboselevel, logger) LOG_LOGGER_IF_ONCE(INFO, IS_VLOG_ON_LOGGER(verboselevel, logger), logger)
#define VLOG_IF_ONCE_LOGGER(verboselevel, condition, logger) LOG_LOGGER_IF_ONCE(INFO, IS_VLOG_ON_LOGGER(verboselevel, logger) && (condition), logger)

/// checks

#define DEFINE_CHECK_OP_IMPL(name, op)                                       \
  template <typename T1, typename T2>                                        \
  inline bool name##Impl(const T1& v1, const T2& v2) {     \
    if (COLLIE_LIKELY(v1 op v2)) {                                     \
      return true;                                                        \
    }                                                                        \
    return false;                              \
  }                                                                          \
  inline bool name##Impl(int v1, int v2) {     \
    return name##Impl<int, int>(v1, v2);                           \
  }

DEFINE_CHECK_OP_IMPL(Check_EQ, ==)

DEFINE_CHECK_OP_IMPL(Check_NE, !=)

DEFINE_CHECK_OP_IMPL(Check_LE, <=)

DEFINE_CHECK_OP_IMPL(Check_LT, <)

DEFINE_CHECK_OP_IMPL(Check_GE, >=)

DEFINE_CHECK_OP_IMPL(Check_GT, >)

#undef DEFINE_CHECK_OP_IMPL

#define CHECK_OP_IMPL_LOGGER(name, op, val1, val2, logger) \
  if(!Check##name##Impl(clog::details::get_referenceable_value(val1), clog::details::get_referenceable_value(val2))) \
    LOG_LOGGER(FATAL, logger) << clog::details::MakeCheckOpString(val1, val2, #val1 " " #op " " #val2)

#define PCHECK_OP_IMPL_LOGGER(name, op, val1, val2, logger) \
  if(!Check##name##Impl(clog::details::get_referenceable_value(val1), clog::details::get_referenceable_value(val2))) \
    PLOG_LOGGER(FATAL, logger) << clog::details::MakeCheckOpString(val1, val2, #val1 " " #op " " #val2)

#define CHECK_OP_LOGGER(name, op, val1, val2, logger) \
  CHECK_OP_IMPL_LOGGER(name, op, val1, val2, logger)

#define PCHECK_OP_LOGGER(name, op, val1, val2, logger) \
  PCHECK_OP_IMPL_LOGGER(name, op, val1, val2, logger)

#define CHECK_LOGGER(condition, logger) \
    LOG_LOGGER_IF(FATAL, COLLIE_UNLIKELY(!(condition)), logger) << "Check failed: " #condition " "
#define CHECK_NOTNULL_LOGGER(val, logger) CHECK_LOGGER(collie::ptr(val) != nullptr, logger)
#define CHECK_PTREQ_LOGGER(val1, val2, logger) CHECK_OP_LOGGER(_EQ, ==, collie::ptr(val1), collie::ptr(val2), logger)
#define CHECK_EQ_LOGGER(val1, val2, logger) CHECK_OP_LOGGER(_EQ, ==, val1, val2, logger)
#define CHECK_NE_LOGGER(val1, val2, logger) CHECK_OP_LOGGER(_NE, !=, val1, val2, logger)
#define CHECK_LE_LOGGER(val1, val2, logger) CHECK_OP_LOGGER(_LE, <=, val1, val2, logger)
#define CHECK_LT_LOGGER(val1, val2, logger) CHECK_OP_LOGGER(_LT, <, val1, val2, logger)
#define CHECK_GE_LOGGER(val1, val2, logger) CHECK_OP_LOGGER(_GE, >=, val1, val2, logger)
#define CHECK_GT_LOGGER(val1, val2, logger) CHECK_OP_LOGGER(_GT, >, val1, val2, logger)
#define CHECK_DOUBLE_EQ_LOGGER(val1, val2, logger)                \
  do {                                             \
    CHECK_LE_LOGGER((val1), (val2) + 0.000000000000001L,logger); \
    CHECK_GE_LOGGER((val1), (val2)-0.000000000000001L,logger);   \
  } while (0)
#define CHECK_NEAR_LOGGER(val1, val2, margin, logger)   \
  do {                                   \
    CHECK_LE_LOGGER((val1), (val2) + (margin), logger); \
    CHECK_GE_LOGGER((val1), (val2) - (margin), logger); \
  } while (0)



#define PCHECK_LOGGER(condition, logger) \
    PLOG_LOGGER_IF(FATAL, COLLIE_UNLIKELY(!(condition)), logger) << "Check failed: " #condition " "
#define PCHECK_PTREQ_LOGGER(val1, val2, logger) PCHECK_OP_LOGGER(_EQ, ==, collie::ptr(val1), collie::ptr(val2), logger)
#define PCHECK_NOTNULL_LOGGER(val, logger) PCHECK_LOGGER(collie::ptr(val) != nullptr, logger)
#define PCHECK_EQ_LOGGER(val1, val2, logger) PCHECK_OP_LOGGER(_EQ, ==, val1, val2, logger)
#define PCHECK_NE_LOGGER(val1, val2, logger) PCHECK_OP_LOGGER(_NE, !=, val1, val2, logger)
#define PCHECK_LE_LOGGER(val1, val2, logger) PCHECK_OP_LOGGER(_LE, <=, val1, val2, logger)
#define PCHECK_LT_LOGGER(val1, val2, logger) PCHECK_OP_LOGGER(_LT, <, val1, val2, logger)
#define PCHECK_GE_LOGGER(val1, val2, logger) PCHECK_OP_LOGGER(_GE, >=, val1, val2, logger)
#define PCHECK_GT_LOGGER(val1, val2, logger) PCHECK_OP_LOGGER(_GT, >, val1, val2, logger)


/// default logger
#define LOG(SEVERITY) LOG_LOGGER(SEVERITY, clog::default_logger_raw())
#define LOG_IF(SEVERITY, condition) LOG_LOGGER_IF(SEVERITY, (condition), clog::default_logger_raw())
#define LOG_EVERY_N(SEVERITY, N) LOG_LOGGER_EVERY_N(SEVERITY, N, clog::default_logger_raw())
#define LOG_IF_EVERY_N(SEVERITY, N, condition) LOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define LOG_FIRST_N(SEVERITY, N) LOG_LOGGER_FIRST_N(SEVERITY, (N), clog::default_logger_raw())
#define LOG_IF_FIRST_N(SEVERITY, N, condition) LOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define LOG_EVERY_T(SEVERITY, seconds) LOG_LOGGER_EVERY_T(SEVERITY, (seconds), clog::default_logger_raw())
#define LOG_IF_EVERY_T(SEVERITY, seconds, condition) LOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), clog::default_logger_raw())
#define LOG_ONCE(SEVERITY) LOG_LOGGER_ONCE(SEVERITY, clog::default_logger_raw())
#define LOG_IF_ONCE(SEVERITY, condition) LOG_LOGGER_IF_ONCE(SEVERITY, (condition), clog::default_logger_raw())

#define CHECK(condition) CHECK_LOGGER(condition, clog::default_logger_raw())
#define CHECK_NOTNULL(val) CHECK(collie::ptr(val) != nullptr)
#define CHECK_EQ(val1, val2) CHECK_OP_LOGGER(_EQ, ==, val1, val2, clog::default_logger_raw())
#define CHECK_NE(val1, val2) CHECK_OP_LOGGER(_NE, !=, val1, val2, clog::default_logger_raw())
#define CHECK_LE(val1, val2) CHECK_OP_LOGGER(_LE, <=, val1, val2, clog::default_logger_raw())
#define CHECK_LT(val1, val2) CHECK_OP_LOGGER(_LT, <, val1, val2, clog::default_logger_raw())
#define CHECK_GE(val1, val2) CHECK_OP_LOGGER(_GE, >=, val1, val2, clog::default_logger_raw())
#define CHECK_GT(val1, val2) CHECK_OP_LOGGER(_GT, >, val1, val2, clog::default_logger_raw())
#define CHECK_DOUBLE_EQ(val1, val2) CHECK_DOUBLE_EQ_LOGGER(val1, val2, clog::default_logger_raw())
#define CHECK_NEAR(val1, val2, margin) CHECK_NEAR_LOGGER(val1, val2, margin, clog::default_logger_raw())
#define CHECK_INDEX(I, A) CHECK(I < (sizeof(A) / sizeof(A[0])))
#define CHECK_BOUND(B, A) CHECK(B <= (sizeof(A) / sizeof(A[0])))

#define PLOG(SEVERITY) PLOG_LOGGER(SEVERITY, clog::default_logger_raw())
#define PLOG_IF(SEVERITY, condition) PLOG_LOGGER_IF(SEVERITY, (condition), clog::default_logger_raw())
#define PLOG_EVERY_N(SEVERITY, N) PLOG_LOGGER_EVERY_N(SEVERITY, N, clog::default_logger_raw())
#define PLOG_IF_EVERY_N(SEVERITY, N, condition) PLOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define PLOG_FIRST_N(SEVERITY, N) PLOG_LOGGER_FIRST_N(SEVERITY, (N), true, clog::default_logger_raw())
#define PLOG_IF_FIRST_N(SEVERITY, N, condition) PLOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define PLOG_EVERY_T(SEVERITY, seconds) PLOG_LOGGER_EVERY_T(SEVERITY, (seconds), clog::default_logger_raw())
#define PLOG_IF_EVERY_T(SEVERITY, seconds, condition) PLOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), clog::default_logger_raw())
#define PLOG_ONCE(SEVERITY) PLOG_LOGGER_ONCE(SEVERITY, clog::default_logger_raw())
#define PLOG_IF_ONCE(SEVERITY, condition) PLOG_LOGGER_IF_ONCE(SEVERITY, (condition), clog::default_logger_raw())

#define PCHECK(condition) PCHECK_LOGGER(condition, clog::default_logger_raw())
#define PCHECK_PTREQ(val1, val2) PCHECK_PTREQ_LOGGER(val1, val2, clog::default_logger_raw())
#define PCHECK_NOTNUL(val) PCHECK_NOTNULL_LOGGER(collie::ptr(val) != nullptr, log::default_logger_raw())
#define PCHECK_EQ(val1, val2) PCHECK_EQ_LOGGER(val1, val2, clog::default_logger_raw())
#define PCHECK_NE(val1, val2) PCHECK_NE_LOGGER(val1, val2, clog::default_logger_raw())
#define PCHECK_LE(val1, val2) PCHECK_LE_LOGGER(val1, val2, clog::default_logger_raw())
#define PCHECK_LT(val1, val2) PCHECK_LT_LOGGER(val1, val2, clog::default_logger_raw())
#define PCHECK_GE(val1, val2) PCHECK_GE_LOGGER(val1, val2, clog::default_logger_raw())
#define PCHECK_GT(val1, val2) PCHECK_GT_LOGGER(val1, val2, clog::default_logger_raw())

#define VLOG(verboselevel) VLOG_LOGGER(verboselevel, clog::default_logger_raw())
#define VLOG_IF(verboselevel, condition) VLOG_IF_LOGGER(verboselevel, (condition), clog::default_logger_raw())
#define VLOG_EVERY_N(verboselevel, N) VLOG_EVERY_N_LOGGER(verboselevel, N, clog::default_logger_raw())
#define VLOG_IF_EVERY_N(verboselevel, N, condition) VLOG_IF_EVERY_N_LOGGER(verboselevel, (N), (condition), clog::default_logger_raw())
#define VLOG_FIRST_N(verboselevel, N) VLOG_FIRST_N_LOGGER(verboselevel, (N), clog::default_logger_raw())
#define VLOG_IF_FIRST_N(verboselevel, N, condition) VLOG_IF_FIRST_N_LOGGER(verboselevel, (N), (condition), clog::default_logger_raw())
#define VLOG_EVERY_T(verboselevel, seconds) VLOG_EVERY_T_LOGGER(verboselevel, (seconds), clog::default_logger_raw())
#define VLOG_IF_EVERY_T(verboselevel, seconds, condition) VLOG_IF_EVERY_T_LOGGER(verboselevel, (seconds), (condition), clog::default_logger_raw())
#define VLOG_ONCE(verboselevel) VLOG_ONCE_LOGGER(verboselevel, clog::default_logger_raw())
#define VLOG_IF_ONCE(verboselevel, condition) VLOG_IF_ONCE_LOGGER(verboselevel, (condition), clog::default_logger_raw())


#if CLOG_DCHECK_IS_ON()

#define DLOG_LOGGER(SEVERITY, logger)                                LOG_LOGGER(SEVERITY, logger)
#define DLOG_LOGGER_IF(SEVERITY, condition, logger)                  LOG_LOGGER_IF(SEVERITY, condition, logger)
#define DLOG_LOGGER_EVERY_N(SEVERITY, N, logger)                     LOG_LOGGER_EVERY_N(SEVERITY, N, logger)
#define DLOG_LOGGER_IF_EVERY_N(SEVERITY, N, condition, logger)       LOG_LOGGER_IF_EVERY_N(SEVERITY, N, condition, logger)
#define DLOG_LOGGER_FIRST_N(SEVERITY, N, condition, logger)          LOG_LOGGER_FIRST_N(SEVERITY, N, condition, logger)
#define DLOG_LOGGER_IF_FIRST_N(SEVERITY, N, condition, logger)       LOG_LOGGER_IF_FIRST_N(SEVERITY, N, condition, logger)
#define DLOG_LOGGER_EVERY_T(SEVERITY, seconds, logger)               LOG_LOGGER_EVERY_T(SEVERITY, seconds, logger)
#define DLOG_LOGGER_IF_EVERY_T(SEVERITY, seconds, condition, logger) LOG_LOGGER_IF_EVERY_T(SEVERITY, seconds, condition, logger)
#define DLOG_LOGGER_ONCE(SEVERITY, logger)                           LOG_LOGGER_ONCE(SEVERITY, logger)
#define DLOG_LOGGER_IF_ONCE(SEVERITY, condition, logger)             LOG_LOGGER_IF_ONCE(SEVERITY, condition, logger)

#define DCHECK_LOGGER(condition, logger) CHECK_LOGGER(condition, logger)
#define DCHECK_NOTNULL_LOGGER(val, logger) CHECK_NOTNULL_LOGGER(val, logger)
#define DCHECK_PTREQ_LOGGER(val1, val2, logger) CHECK_PTREQ_LOGGER(val1, val2, logger)
#define DCHECK_EQ_LOGGER(val1, val2, logger) CHECK_EQ_LOGGER(val1, val2, logger)
#define DCHECK_NE_LOGGER(val1, val2, logger) CHECK_NE_LOGGER(val1, val2,logger)
#define DCHECK_LE_LOGGER(val1, val2, logger) CHECK_LE_LOGGER(val1, val2,logger)
#define DCHECK_LT_LOGGER(val1, val2, logger) CHECK_LT_LOGGER(val1, val2,logger)
#define DCHECK_GE_LOGGER(val1, val2, logger) CHECK_GE_LOGGER(val1, val2,logger)
#define DCHECK_GT_LOGGER(val1, val2, logger) CHECK_GT_LOGGER(val1, val2,logger)

#define DPLOG_LOGGER(SEVERITY, logger)                                PLOG_LOGGER(SEVERITY, logger)
#define DPLOG_LOGGER_IF(SEVERITY, condition, logger)                  PLOG_LOGGER_IF(SEVERITY, condition, logger)
#define DPLOG_LOGGER_EVERY_N(SEVERITY, N, logger)                     PLOG_LOGGER_EVERY_N(SEVERITY, N, logger)
#define DPLOG_LOGGER_IF_EVERY_N(SEVERITY, N, condition, logger)       PLOG_LOGGER_IF_EVERY_N(SEVERITY, N, condition, logger)
#define DPLOG_LOGGER_FIRST_N(SEVERITY, N, condition, logger)          PLOG_LOGGER_FIRST_N(SEVERITY, N, condition, logger)
#define DPLOG_LOGGER_IF_FIRST_N(SEVERITY, N, condition, logger)       PLOG_LOGGER_IF_FIRST_N(SEVERITY, N, condition, logger)
#define DPLOG_LOGGER_EVERY_T(SEVERITY, seconds, logger)               PLOG_LOGGER_EVERY_T(SEVERITY, seconds, logger)
#define DPLOG_LOGGER_IF_EVERY_T(SEVERITY, seconds, condition, logger) PLOG_LOGGER_IF_EVERY_T(SEVERITY, seconds, condition, logger)
#define DPLOG_LOGGER_ONCE(SEVERITY, logger)                           PLOG_LOGGER_ONCE(SEVERITY, logger)
#define DPLOG_LOGGER_IF_ONCE(SEVERITY, condition, logger)             PLOG_LOGGER_IF_ONCE(SEVERITY, condition, logger)

#define DPCHECK_LOGGER(condition, logger) PCHECK_LOGGER(condition, logger)
#define DPCHECK_NOTNULL_LOGGER(val, logger) PCHECK_NOTNULL_LOGGER(val, logger)
#define DPCHECK_PTREQ_LOGGER(val1, val2, logger) PCHECK_PTREQ_LOGGER(val1, val2, logger)
#define DPCHECK_EQ_LOGGER(val1, val2, logger) PCHECK_EQ_LOGGER(val1, val2, logger)
#define DPCHECK_NE_LOGGER(val1, val2, logger) PCHECK_NE_LOGGER(val1, val2, logger)
#define DPCHECK_LE_LOGGER(val1, val2, logger) PCHECK_LE_LOGGER(val1, val2, logger)
#define DPCHECK_LT_LOGGER(val1, val2, logger) PCHECK_LT_LOGGER(val1, val2, logger)
#define DPCHECK_GE_LOGGER(val1, val2, logger) PCHECK_GE_LOGGER(val1, val2, logger)
#define DPCHECK_GT_LOGGER(val1, val2, logger) PCHECK_GT_LOGGER(val1, val2, logger)

#define DVLOG_LOGGER(verboselevel, logger)  VLOG_LOGGER(verboselevel, logger)
#define DVLOG_IF_LOGGER(verboselevel, condition, logger) VLOG_IF_LOGGER(verboselevel, condition, logger)
#define DVLOG_EVERY_N_LOGGER(verboselevel, N, logger) VLOG_EVERY_N_LOGGER(verboselevel, N, logger)
#define DVLOG_IF_EVERY_N_LOGGER(verboselevel, N, condition, logger) VLOG_IF_EVERY_N_LOGGER(verboselevel, N, condition, logger)
#define DVLOG_FIRST_N_LOGGER(verboselevel, N, logger) VLOG_FIRST_N_LOGGER(verboselevel, N, logger)
#define DVLOG_IF_FIRST_N_LOGGER(verboselevel, N, condition, logger) VLOG_IF_FIRST_N_LOGGER(verboselevel, N, condition, logger)
#define DVLOG_EVERY_T_LOGGER(verboselevel, seconds, logger) VLOG_EVERY_T_LOGGER(verboselevel, seconds, logger)
#define DVLOG_IF_EVERY_T_LOGGER(verboselevel, seconds, condition, logger) VLOG_IF_EVERY_T_LOGGER(verboselevel, seconds, condition, logger)
#define DVLOG_ONCE_LOGGER(verboselevel, logger) VLOG_ONCE_LOGGER(verboselevel, logger)
#define DVLOG_IF_ONCE_LOGGER(verboselevel, condition, logger) VLOG_IF_ONCE_LOGGER(verboselevel, condition, logger)

#define DLOG(SEVERITY)                                LOG(SEVERITY)
#define DLOG_IF(SEVERITY, condition)                  LOG_LOGGER_IF(SEVERITY, (condition), clog::default_logger_raw())
#define DLOG_EVERY_N(SEVERITY, N)                     LOG_LOGGER_EVERY_N(SEVERITY, N, clog::default_logger_raw())
#define DLOG_IF_EVERY_N(SEVERITY, N, condition)       LOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define DLOG_FIRST_N(SEVERITY, N)                     LOG_LOGGER_FIRST_N(SEVERITY, (N), true, clog::default_logger_raw())
#define DLOG_IF_FIRST_N(SEVERITY, N, condition)       LOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define DLOG_EVERY_T(SEVERITY, seconds)               LOG_LOGGER_EVERY_T(SEVERITY, (seconds), clog::default_logger_raw())
#define DLOG_IF_EVERY_T(SEVERITY, seconds, condition) LOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), clog::default_logger_raw())
#define DLOG_ONCE(SEVERITY)                           LOG_LOGGER_ONCE(SEVERITY, clog::default_logger_raw())
#define DLOG_IF_ONCE(SEVERITY, condition)             LOG_LOGGER_IF_ONCE(SEVERITY, (condition), clog::default_logger_raw())

#define DCHECK(condition) DCHECK_LOGGER(condition, clog::default_logger_raw())
#define DCHECK_NOTNULL(val) DCHECK_NOTNULL_LOGGER(val, clog::default_logger_raw())
#define DCHECK_PTREQ(val1, val2) DCHECK_PTREQ_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_EQ(val1, val2) CHECK_EQ_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_NE(val1, val2) CHECK_NE_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_LE(val1, val2) CHECK_LE_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_LT(val1, val2) CHECK_LT_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_GE(val1, val2) CHECK_GE_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_GT(val1, val2) CHECK_GT_LOGGER(val1, val2, clog::default_logger_raw())

#define DPLOG(SEVERITY)                                  PLOG_LOGGER(SEVERITY, clog::default_logger_raw())
#define DPLOG_IF(SEVERITY, condition)                    PLOG_LOGGER_IF(SEVERITY, (condition), clog::default_logger_raw())
#define DPLOG_EVERY_N(SEVERITY, N)                       PLOG_LOGGER_EVERY_N(SEVERITY, N, clog::default_logger_raw())
#define DPLOG_IF_EVERY_N(SEVERITY, N, condition)         PLOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define DPLOG_FIRST_N(SEVERITY, N)                       PLOG_LOGGER_FIRST_N(SEVERITY, (N), true, clog::default_logger_raw())
#define DPLOG_IF_FIRST_N(SEVERITY, N, condition)         PLOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define DPLOG_EVERY_T(SEVERITY, seconds)                 PLOG_LOGGER_EVERY_T(SEVERITY, (seconds), clog::default_logger_raw())
#define DPLOG_IF_EVERY_T(SEVERITY, seconds, condition)   PLOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), clog::default_logger_raw())
#define DPLOG_ONCE(SEVERITY)                             PLOG_LOGGER_ONCE(SEVERITY, clog::default_logger_raw())
#define DPLOG_IF_ONCE(SEVERITY, condition)               PLOG_LOGGER_IF_ONCE(SEVERITY, (condition), clog::default_logger_raw())

#define DPCHECK(condition) DPCHECK_LOGGER(condition, clog::default_logger_raw())
#define DPCHECK_NOTNULL(val) DPCHECK_NOTNULL_LOGGER(val, clog::default_logger_raw())
#define DPCHECK_PTREQ(val1, val2) DPCHECK_PTREQ_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_EQ(val1, val2) PCHECK_EQ_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_NE(val1, val2) PCHECK_NE_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_LE(val1, val2) PCHECK_LE_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_LT(val1, val2) PCHECK_LT_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_GE(val1, val2) PCHECK_GE_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_GT(val1, val2) PCHECK_GT_LOGGER(val1, val2, clog::default_logger_raw())

#define DVLOG(verboselevel) DVLOG_LOGGER(verboselevel, clog::default_logger_raw())
#define DVLOG_IF(verboselevel, condition) DVLOG_IF_LOGGER(verboselevel, (condition), clog::default_logger_raw())
#define DVLOG_EVERY_N(verboselevel, N) DVLOG_EVERY_N_LOGGER(verboselevel, N, clog::default_logger_raw())
#define DVLOG_IF_EVERY_N(verboselevel, N, condition) DVLOG_IF_EVERY_N_LOGGER(verboselevel, (N), (condition), clog::default_logger_raw())
#define DVLOG_FIRST_N(verboselevel, N) DVLOG_FIRST_N_LOGGER(verboselevel, (N), clog::default_logger_raw())
#define DVLOG_IF_FIRST_N(verboselevel, N, condition) DVLOG_IF_FIRST_N_LOGGER(verboselevel, (N), (condition), clog::default_logger_raw())
#define DVLOG_EVERY_T(verboselevel, seconds) DVLOG_EVERY_T_LOGGER(verboselevel, (seconds), clog::default_logger_raw())
#define DVLOG_IF_EVERY_T(verboselevel, seconds, condition) DVLOG_IF_EVERY_T_LOGGER(verboselevel, (seconds), (condition), clog::default_logger_raw())
#define DVLOG_ONCE(verboselevel) DVLOG_ONCE_LOGGER(verboselevel, clog::default_logger_raw())
#define DVLOG_IF_ONCE(verboselevel, condition) DVLOG_IF_ONCE_LOGGER(verboselevel, (condition), clog::default_logger_raw())

#else // NDEBUG

#define DLOG_LOGGER(SEVERITY, logger)                                \
    static_cast<void>(0), \
            true ? (void)0    \
                 : LOG_LOGGER(SEVERITY, logger)

#define DLOG_LOGGER_IF(SEVERITY, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0    \
                 : LOG_LOGGER(SEVERITY, logger)

#define DLOG_LOGGER_EVERY_N(SEVERITY, N, logger) \
    static_cast<void>(0), \
            true ? (void)0    \
                 : LOG_LOGGER_EVERY_N(SEVERITY, N, logger)

#define DLOG_LOGGER_IF_EVERY_N(SEVERITY, N, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0    \
            :LOG_LOGGER_EVERY_N(SEVERITY, N, logger)

#define DLOG_LOGGER_FIRST_N(SEVERITY, N, condition, logger) \
    static_cast<void>(0), \
            true ? (void)0    \
                 : LOG_LOGGER_FIRST_N(SEVERITY, N, logger)

#define DLOG_LOGGER_IF_FIRST_N(SEVERITY, N, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0    \
                 : LOG_LOGGER_FIRST_N(SEVERITY, N, logger)


#define DLOG_LOGGER_EVERY_T(SEVERITY, seconds, logger) \
    static_cast<void>(0), \
            true ? (void)0    \
                 : LOG_LOGGER_EVERY_T(SEVERITY, seconds, logger)

#define DLOG_LOGGER_IF_EVERY_T(SEVERITY, seconds, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0                         \
            : LOG_LOGGER_EVERY_T(SEVERITY, seconds, logger)

#define DLOG_LOGGER_ONCE(SEVERITY, logger) \
    static_cast<void>(0), \
            true ? (void)0                 \
                 : LOG_LOGGER_ONCE(SEVERITY, logger)

#define DLOG_LOGGER_IF_ONCE(SEVERITY, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0             \
                    : LOG_LOGGER_ONCE(SEVERITY, logger)

#define DPLOG_LOGGER(SEVERITY, logger) \
static_cast<void>(0), \
            true ? (void)0    \
                 : PLOG_LOGGER(SEVERITY, logger)

#define DPLOG_LOGGER_IF(SEVERITY, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0         \
            : PLOG_LOGGER(SEVERITY, logger)


#define DPLOG_LOGGER_EVERY_N(SEVERITY, N, logger) \
    static_cast<void>(0), \
            true ? (void)0    \
                 : PLOG_LOGGER_EVERY_N(SEVERITY, N, logger)

#define DPLOG_LOGGER_IF_EVERY_N(SEVERITY, N, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0    \
            :PLOG_LOGGER_EVERY_N(SEVERITY, N, logger)

#define DPLOG_LOGGER_FIRST_N(SEVERITY, N, condition, logger) \
static_cast<void>(0), \
            true ? (void)0    \
                 : PLOG_LOGGER_FIRST_N(SEVERITY, N, logger)

#define DPLOG_LOGGER_IF_FIRST_N(SEVERITY, N, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0    \
                 : PLOG_LOGGER_FIRST_N(SEVERITY, N, logger)

#define DPLOG_LOGGER_EVERY_T(SEVERITY, seconds, logger) \
    static_cast<void>(0), \
            true ? (void)0    \
                 : PLOG_LOGGER_EVERY_T(SEVERITY, seconds, logger)

#define DPLOG_LOGGER_IF_EVERY_T(SEVERITY, seconds, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0                         \
            : PLOG_LOGGER_EVERY_T(SEVERITY, seconds, logger)

#define DPLOG_LOGGER_ONCE(SEVERITY, logger) \
    static_cast<void>(0), \
            true ? (void)0                 \
                 : PLOG_LOGGER_ONCE(SEVERITY, logger)
#define DPLOG_LOGGER_IF_ONCE(SEVERITY, condition, logger) \
    static_cast<void>(0), \
            (true && !(condition)) ? (void)0             \
                    : PLOG_LOGGER_ONCE(SEVERITY, logger)

#define DVLOG_LOGGER(verboselevel, logger) \
    static_cast<void>(0), \
            true ? (void)0    \
                 : VLOG_LOGGER(verboselevel, logger)

#define DVLOG_IF_LOGGER(verboselevel, condition, logger) \
    static_cast<void>(0), \
            (true) ? (void)0    \
                 : VLOG_IF_LOGGER(verboselevel, condition, logger)

#define DVLOG_EVERY_N_LOGGER(verboselevel, N, logger) \
    static_cast<void>(0), \
            true ? (void)    \
                 : VLOG_EVERY_N_LOGGER(verboselevel, N, logger)

#define DVLOG_IF_EVERY_N_LOGGER(verboselevel, N, condition, logger) \
    static_cast<void>(0), \
            (true) ? (void)0    \
            :VLOG_IF_EVERY_N_LOGGER(verboselevel, N, condition, logger)

#define DVLOG_FIRST_N_LOGGER(verboselevel, N, logger) \
    static_cast<void>(0), \
            true ? (void)    \
                 : VLOG_FIRST_N_LOGGER(verboselevel, N, logger)

#define DVLOG_IF_FIRST_N_LOGGER(verboselevel, N, condition, logger) \
    static_cast<void>(0), \
            (true) ? (void)0    \
                 : VLOG_IF_FIRST_N_LOGGER(verboselevel, N, condition, logger)

#define DVLOG_EVERY_T_LOGGER(verboselevel, seconds, logger) \
    static_cast<void>(0), \
            true ? (void)    \
                 : VLOG_EVERY_T_LOGGER(verboselevel, seconds, logger)

#define DVLOG_IF_EVERY_T_LOGGER(verboselevel, seconds, condition, logger) \
    static_cast<void>(0), \
            (true) ? (void)0    \
            :VLOG_IF_EVERY_T_LOGGER(verboselevel, seconds, condition, logger)

#define DVLOG_ONCE_LOGGER(verboselevel, logger) \
    static_cast<void>(0), \
            true ? (void)    \
                 : VLOG_ONCE_LOGGER(verboselevel, logger)

#define DVLOG_IF_ONCE_LOGGER(verboselevel, condition, logger) \
    static_cast<void>(0), \
            (true) ? (void)0    \
                 : VLOG_IF_ONCE_LOGGER(verboselevel, condition, logger)

#define DCHECK_LOGGER(condition, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() CHECK_LOGGER(condition, logger)

#define DCHECK_NOTNULL_LOGGER(val, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
        while(false) COLLIE_MSVC_POP_WARNING() CHECK_NOTNULL_LOGGER(val, logger)

#define DCHECK_PTREQ_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
        while(false) COLLIE_MSVC_POP_WARNING() CHECK_PTREQ_LOGGER(val1, val2, logger)

//CHECK_OP_LOGGER(_EQ, ==, collie::ptr(val1), collie::ptr(val2), logger)

#define DCHECK_EQ_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() CHECK_EQ_LOGGER(val1, val2, logger)

#define DCHECK_NE_LOGGER(val1, val2, logger) \
COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() CHECK_NE_LOGGER(val1, val2,logger)

#define DCHECK_LE_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() CHECK_LE_LOGGER(val1, val2,logger)

#define DCHECK_LT_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() CHECK_LT_LOGGER(val1, val2,logger)

#define DCHECK_GE_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() CHECK_GE_LOGGER(val1, val2,logger)

#define DCHECK_GT_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() CHECK_GT_LOGGER(val1, val2,logger)

#define DPCHECK_LOGGER(condition, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() PCHECK_LOGGER(condition, logger)

#define DPCHECK_NOTNULL_LOGGER(val, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
        while(false) COLLIE_MSVC_POP_WARNING() PCHECK_NOTNULL_LOGGER(val, logger)

#define DPCHECK_PTREQ_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
        while(false) COLLIE_MSVC_POP_WARNING() PCHECK_PTREQ_LOGGER(val1, val2, logger)

#define DPCHECK_EQ_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() PCHECK_EQ_LOGGER(val1, val2, logger)

#define DPCHECK_NE_LOGGER(val1, val2, logger) \
    COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() PCHECK_NE_LOGGER(val1, val2, logger)

#define DPCHECK_LE_LOGGER(val1, val2, logger) \
COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() PCHECK_LE_LOGGER(val1, val2, logger)

#define DPCHECK_LT_LOGGER(val1, val2, logger) \
COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() PCHECK_LT_LOGGER(val1, val2, logger)

#define DPCHECK_GE_LOGGER(val1, val2, logger) \
COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() PCHECK_GE_LOGGER(val1, val2, logger)

#define DPCHECK_GT_LOGGER(val1, val2, logger) \
COLLIE_MSVC_PUSH_DISABLE_WARNING(4127)                      \
    while(false) COLLIE_MSVC_POP_WARNING() PCHECK_GT_LOGGER(val1, val2, logger)

#define DLOG(SEVERITY)                                DLOG_LOGGER(SEVERITY, clog::default_logger_raw())
#define DLOG_IF(SEVERITY, condition)                  DLOG_LOGGER_IF(SEVERITY, (condition), clog::default_logger_raw())
#define DLOG_EVERY_N(SEVERITY, N)                     DLOG_LOGGER_EVERY_N(SEVERITY, N, clog::default_logger_raw())
#define DLOG_IF_EVERY_N(SEVERITY, N, condition)       DLOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define DLOG_FIRST_N(SEVERITY, N)                     DLOG_LOGGER_FIRST_N(SEVERITY, (N), true, clog::default_logger_raw())
#define DLOG_IF_FIRST_N(SEVERITY, N, condition)       DLOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define DLOG_EVERY_T(SEVERITY, seconds)               DLOG_LOGGER_EVERY_T(SEVERITY, (seconds), clog::default_logger_raw())
#define DLOG_IF_EVERY_T(SEVERITY, seconds, condition) DLOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), clog::default_logger_raw())
#define DLOG_ONCE(SEVERITY)                           DLOG_LOGGER_ONCE(SEVERITY, clog::default_logger_raw())
#define DLOG_IF_ONCE(SEVERITY, condition)             DLOG_LOGGER_IF_ONCE(SEVERITY, (condition), clog::default_logger_raw())

#define DCHECK(condition) DCHECK_LOGGER(condition, clog::default_logger_raw())
#define DCHECK_NOTNULL(val) DCHECK_NOTNULL_LOGGER(val, clog::default_logger_raw())
#define DCHECK_PTREQ(val1, val2) DCHECK_PTREQ_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_EQ(val1, val2) CHECK_EQ_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_NE(val1, val2) CHECK_NE_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_LE(val1, val2) CHECK_LE_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_LT(val1, val2) CHECK_LT_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_GE(val1, val2) CHECK_GE_LOGGER(val1, val2, clog::default_logger_raw())
#define DCHECK_GT(val1, val2) CHECK_GT_LOGGER(val1, val2, clog::default_logger_raw())


#define DPLOG(SEVERITY)                                  DPLOG_LOGGER(SEVERITY, clog::default_logger_raw())
#define DPLOG_IF(SEVERITY, condition)                    DPLOG_LOGGER_IF(SEVERITY, (condition), clog::default_logger_raw())
#define DPLOG_EVERY_N(SEVERITY, N)                       DPLOG_LOGGER_EVERY_N(SEVERITY, N, clog::default_logger_raw())
#define DPLOG_IF_EVERY_N(SEVERITY, N, condition)         DPLOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define DPLOG_FIRST_N(SEVERITY, N)                       DPLOG_LOGGER_FIRST_N(SEVERITY, (N), true, clog::default_logger_raw())
#define DPLOG_IF_FIRST_N(SEVERITY, N, condition)         DPLOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), clog::default_logger_raw())
#define DPLOG_EVERY_T(SEVERITY, seconds)                 DPLOG_LOGGER_EVERY_T(SEVERITY, (seconds), clog::default_logger_raw())
#define DPLOG_IF_EVERY_T(SEVERITY, seconds, condition)   DPLOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), clog::default_logger_raw())
#define DPLOG_ONCE(SEVERITY)                             DPLOG_LOGGER_ONCE(SEVERITY, clog::default_logger_raw())
#define DPLOG_IF_ONCE(SEVERITY, condition)               DPLOG_LOGGER_IF_ONCE(SEVERITY, (condition), clog::default_logger_raw())

#define DPCHECK(condition)        DPCHECK_LOGGER(condition, clog::default_logger_raw())
#define DPCHECK_NOTNULL(val)      DPCHECK_NOTNULL_LOGGER(val, clog::default_logger_raw())
#define DPCHECK_PTREQ(val1, val2) DPCHECK_PTREQ_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_EQ(val1, val2) DPCHECK_EQ_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_NE(val1, val2) DPCHECK_NE_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_LE(val1, val2) DPCHECK_LE_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_LT(val1, val2) DPCHECK_LT_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_GE(val1, val2) DPCHECK_GE_LOGGER(val1, val2, clog::default_logger_raw())
#define DPCHECK_GT(val1, val2) DPCHECK_GT_LOGGER(val1, val2, clog::default_logger_raw())

#define DVLOG(verboselevel) DVLOG_LOGGER(verboselevel, clog::default_logger_raw())
#define DVLOG_IF(verboselevel, condition) DVLOG_IF_LOGGER(verboselevel, (condition), clog::default_logger_raw())
#define DVLOG_EVERY_N(verboselevel, N) DVLOG_EVERY_N_LOGGER(verboselevel, N, clog::default_logger_raw())
#define DVLOG_IF_EVERY_N(verboselevel, N, condition) DVLOG_IF_EVERY_N_LOGGER(verboselevel, (N), (condition), clog::default_logger_raw())
#define DVLOG_FIRST_N(verboselevel, N) DVLOG_FIRST_N_LOGGER(verboselevel, (N), clog::default_logger_raw())
#define DVLOG_IF_FIRST_N(verboselevel, N, condition) DVLOG_IF_FIRST_N_LOGGER(verboselevel, (N), (condition), clog::default_logger_raw())
#define DVLOG_EVERY_T(verboselevel, seconds) DVLOG_EVERY_T_LOGGER(verboselevel, (seconds), clog::default_logger_raw())
#define DVLOG_IF_EVERY_T(verboselevel, seconds, condition) DVLOG_IF_EVERY_T_LOGGER(verboselevel, (seconds), (condition), clog::default_logger_raw())
#define DVLOG_ONCE(verboselevel) DVLOG_ONCE_LOGGER(verboselevel, clog::default_logger_raw())
#define DVLOG_IF_ONCE(verboselevel, condition) DVLOG_IF_ONCE_LOGGER(verboselevel, (condition), clog::default_logger_raw())

#endif
