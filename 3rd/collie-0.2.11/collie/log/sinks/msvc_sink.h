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

#if defined(_WIN32)

    #include <collie/log/details/null_mutex.h>
    #if defined(CLOG_WCHAR_TO_UTF8_SUPPORT)
        #include <collie/log/details/os.h>
    #endif
    #include <collie/log/sinks/base_sink.h>

    #include <mutex>
    #include <string>

    // Avoid including windows.h (https://stackoverflow.com/a/30741042)
    #if defined(CLOG_WCHAR_TO_UTF8_SUPPORT)
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringW(const wchar_t *lpOutputString);
    #else
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(const char *lpOutputString);
    #endif
extern "C" __declspec(dllimport) int __stdcall IsDebuggerPresent();

namespace clog {
namespace sinks {
/*
 * MSVC sink (logging using OutputDebugStringA)
 */
template <typename Mutex>
class msvc_sink : public base_sink<Mutex> {
public:
    msvc_sink() = default;
    msvc_sink(bool check_debugger_present)
        : check_debugger_present_{check_debugger_present} {};

protected:
    void sink_it_(const details::log_msg &msg) override {
        if (check_debugger_present_ && !IsDebuggerPresent()) {
            return;
        }
        memory_buf_t formatted;
        base_sink<Mutex>::formatter_->format(msg, formatted);
        formatted.push_back('\0');  // add a null terminator for OutputDebugString
    #if defined(CLOG_WCHAR_TO_UTF8_SUPPORT)
        wmemory_buf_t wformatted;
        details::os::utf8_to_wstrbuf(string_view_t(formatted.data(), formatted.size()), wformatted);
        OutputDebugStringW(wformatted.data());
    #else
        OutputDebugStringA(formatted.data());
    #endif
    }

    void flush_() override {}

    bool check_debugger_present_ = true;
};

using msvc_sink_mt = msvc_sink<std::mutex>;
using msvc_sink_st = msvc_sink<details::null_mutex>;

using windebug_sink_mt = msvc_sink_mt;
using windebug_sink_st = msvc_sink_st;

}  // namespace sinks
}  // namespace clog

#endif
