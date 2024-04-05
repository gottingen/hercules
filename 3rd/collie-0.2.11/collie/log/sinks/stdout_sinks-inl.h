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

#include <memory>
#include <collie/log/details/console_globals.h>
#include <collie/log/pattern_formatter.h>

#ifdef _WIN32
// under windows using fwrite to non-binary stream results in \r\r\n (see issue #1675)
// so instead we use ::FileWrite
#include <collie/log/details/windows_include.h>

#ifndef _USING_V110_SDK71_  // fileapi.h doesn't exist in winxp
#include <fileapi.h>    // WriteFile (..)
#endif

#include <io.h>     // _get_osfhandle(..)
#include <stdio.h>  // _fileno(..)
#endif                  // WIN32

namespace clog {

    namespace sinks {

        template<typename ConsoleMutex>
        inline stdout_sink_base<ConsoleMutex>::stdout_sink_base(FILE *file)
                : mutex_(ConsoleMutex::mutex()),
                  file_(file),
                  formatter_(details::make_unique<clog::pattern_formatter>()) {
#ifdef _WIN32
            // get windows handle from the FILE* object

            handle_ = reinterpret_cast<HANDLE>(::_get_osfhandle(::_fileno(file_)));

            // don't throw to support cases where no console is attached,
            // and let the log method to do nothing if (handle_ == INVALID_HANDLE_VALUE).
            // throw only if non stdout/stderr target is requested (probably regular file and not console).
            if (handle_ == INVALID_HANDLE_VALUE && file != stdout && file != stderr) {
                throw_clog_ex("clog::stdout_sink_base: _get_osfhandle() failed", errno);
            }
#endif  // WIN32
        }

        template<typename ConsoleMutex>
        inline void stdout_sink_base<ConsoleMutex>::log(const details::log_msg &msg) {
#ifdef _WIN32
            if (handle_ == INVALID_HANDLE_VALUE) {
                return;
            }
            std::lock_guard<mutex_t> lock(mutex_);
            memory_buf_t formatted;
            formatter_->format(msg, formatted);
            auto size = static_cast<DWORD>(formatted.size());
            DWORD bytes_written = 0;
            bool ok = ::WriteFile(handle_, formatted.data(), size, &bytes_written, nullptr) != 0;
            if (!ok) {
                throw_clog_ex("stdout_sink_base: WriteFile() failed. GetLastError(): " +
                                std::to_string(::GetLastError()));
            }
#else
            std::lock_guard<mutex_t> lock(mutex_);
            memory_buf_t formatted;
            formatter_->format(msg, formatted);
            ::fwrite(formatted.data(), sizeof(char), formatted.size(), file_);
#endif                // WIN32
            ::fflush(file_);  // flush every line to terminal
        }

        template<typename ConsoleMutex>
        inline void stdout_sink_base<ConsoleMutex>::flush() {
            std::lock_guard<mutex_t> lock(mutex_);
            fflush(file_);
        }

        template<typename ConsoleMutex>
        inline void stdout_sink_base<ConsoleMutex>::set_pattern(const std::string &pattern) {
            std::lock_guard<mutex_t> lock(mutex_);
            formatter_ = std::unique_ptr<clog::formatter>(new pattern_formatter(pattern));
        }

        template<typename ConsoleMutex>
        inline void stdout_sink_base<ConsoleMutex>::set_formatter(
                std::unique_ptr<clog::formatter> sink_formatter) {
            std::lock_guard<mutex_t> lock(mutex_);
            formatter_ = std::move(sink_formatter);
        }

// stdout sink
        template<typename ConsoleMutex>
        inline stdout_sink<ConsoleMutex>::stdout_sink()
                : stdout_sink_base<ConsoleMutex>(stdout) {}

// stderr sink
        template<typename ConsoleMutex>
        inline stderr_sink<ConsoleMutex>::stderr_sink()
                : stdout_sink_base<ConsoleMutex>(stderr) {}

    }  // namespace sinks

// factory methods
    template<typename Factory>
    inline std::shared_ptr<logger> stdout_logger_mt(const std::string &logger_name) {
        return Factory::template create<sinks::stdout_sink_mt>(logger_name);
    }

    template<typename Factory>
    inline std::shared_ptr<logger> stdout_logger_st(const std::string &logger_name) {
        return Factory::template create<sinks::stdout_sink_st>(logger_name);
    }

    template<typename Factory>
    inline std::shared_ptr<logger> stderr_logger_mt(const std::string &logger_name) {
        return Factory::template create<sinks::stderr_sink_mt>(logger_name);
    }

    template<typename Factory>
    inline std::shared_ptr<logger> stderr_logger_st(const std::string &logger_name) {
        return Factory::template create<sinks::stderr_sink_st>(logger_name);
    }
}  // namespace clog