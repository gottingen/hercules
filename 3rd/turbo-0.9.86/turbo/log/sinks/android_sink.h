// Copyright 2023 The titan-search Authors.
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

#pragma once

#ifdef __ANDROID__

#    include <turbo/log/details/fmt_helper.h>
#    include <turbo/log/details/null_mutex.h>
#    include <turbo/log/details/os.h>
#    include <turbo/log/sinks/base_sink.h>
#    include <turbo/log/details/synchronous_factory.h>

#    include <android/log.h>
#    include <chrono>
#    include <mutex>
#    include <string>
#    include <thread>
#    include <type_traits>

#    if !defined(TLOG_ANDROID_RETRIES)
#        define TLOG_ANDROID_RETRIES 2
#    endif
#include "turbo/times/clock.h"
namespace turbo::tlog {
namespace sinks {

/*
 * Android sink
 * (logging using __android_log_write or __android_log_buf_write depending on the specified BufferID)
 */
template<typename Mutex, int BufferID = log_id::LOG_ID_MAIN>
class android_sink final : public base_sink<Mutex>
{
public:
    explicit android_sink(std::string tag = "tlog", bool use_raw_msg = false)
        : tag_(std::move(tag))
        , use_raw_msg_(use_raw_msg)
    {}

protected:
    void sink_it_(const details::log_msg &msg) override
    {
        const android_LogPriority priority = convert_to_android_(msg.level);
        memory_buf_t formatted;
        if (use_raw_msg_)
        {
            details::fmt_helper::append_string_view(msg.payload, formatted);
        }
        else
        {
            base_sink<Mutex>::formatter_->format(msg, formatted);
        }
        formatted.push_back('\0');
        const char *msg_output = formatted.data();

        // See system/core/liblog/logger_write.c for explanation of return value
        int ret = android_log(priority, tag_.c_str(), msg_output);
        int retry_count = 0;
        while ((ret == -11 /*EAGAIN*/) && (retry_count < TLOG_ANDROID_RETRIES))
        {
            turbo::turbo::sleep_for(turbo::Duration::milliseconds(5));
            ret = android_log(priority, tag_.c_str(), msg_output);
            retry_count++;
        }

        if (ret < 0)
        {
            throw_tlog_ex("logging to Android failed", ret);
        }
    }

    void flush_() override {}

private:
    // There might be liblog versions used, that do not support __android_log_buf_write. So we only compile and link against
    // __android_log_buf_write, if user explicitely provides a non-default log buffer. Otherwise, when using the default log buffer, always
    // log via __android_log_write.
    template<int ID = BufferID>
    typename std::enable_if<ID == static_cast<int>(log_id::LOG_ID_MAIN), int>::type android_log(int prio, const char *tag, const char *text)
    {
        return __android_log_write(prio, tag, text);
    }

    template<int ID = BufferID>
    typename std::enable_if<ID != static_cast<int>(log_id::LOG_ID_MAIN), int>::type android_log(int prio, const char *tag, const char *text)
    {
        return __android_log_buf_write(ID, prio, tag, text);
    }

    static android_LogPriority convert_to_android_(turbo::tlog::level::level_enum level)
    {
        switch (level)
        {
        case turbo::tlog::level::trace:
            return ANDROID_LOG_VERBOSE;
        case turbo::tlog::level::debug:
            return ANDROID_LOG_DEBUG;
        case turbo::tlog::level::info:
            return ANDROID_LOG_INFO;
        case turbo::tlog::level::warn:
            return ANDROID_LOG_WARN;
        case turbo::tlog::level::err:
            return ANDROID_LOG_ERROR;
        case turbo::tlog::level::critical:
            return ANDROID_LOG_FATAL;
        default:
            return ANDROID_LOG_DEFAULT;
        }
    }

    std::string tag_;
    bool use_raw_msg_;
};

using android_sink_mt = android_sink<std::mutex>;
using android_sink_st = android_sink<details::null_mutex>;

template<int BufferId = log_id::LOG_ID_MAIN>
using android_sink_buf_mt = android_sink<std::mutex, BufferId>;
template<int BufferId = log_id::LOG_ID_MAIN>
using android_sink_buf_st = android_sink<details::null_mutex, BufferId>;

} // namespace sinks

// Create and register android syslog logger

template<typename Factory = turbo::tlog::synchronous_factory>
inline std::shared_ptr<logger> android_logger_mt(const std::string &logger_name, const std::string &tag = "tlog")
{
    return Factory::template create<sinks::android_sink_mt>(logger_name, tag);
}

template<typename Factory = turbo::tlog::synchronous_factory>
inline std::shared_ptr<logger> android_logger_st(const std::string &logger_name, const std::string &tag = "tlog")
{
    return Factory::template create<sinks::android_sink_st>(logger_name, tag);
}

} // namespace turbo::tlog

#endif // __ANDROID__