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

#include "turbo/log/common.h"
#include <ctime> // std::time_t

namespace turbo::tlog {
namespace details {
namespace os {

//TURBO_DLL turbo::tlog::log_clock::time_point now() noexcept;
/*
TURBO_DLL std::tm localtime(const std::time_t &time_tt) noexcept;

TURBO_DLL std::tm localtime() noexcept;

TURBO_DLL std::tm gmtime(const std::time_t &time_tt) noexcept;

TURBO_DLL std::tm gmtime() noexcept;
*/
// eol definition
#if !defined(TLOG_EOL)
#    ifdef _WIN32
#        define TLOG_EOL "\r\n"
#    else
#        define TLOG_EOL "\n"
#    endif
#endif

constexpr static const char *default_eol = TLOG_EOL;

// folder separator
#if !defined(TLOG_FOLDER_SEPS)
#    ifdef _WIN32
#        define TLOG_FOLDER_SEPS "\\/"
#    else
#        define TLOG_FOLDER_SEPS "/"
#    endif
#endif

constexpr static const char folder_seps[] = TLOG_FOLDER_SEPS;
constexpr static const filename_t::value_type folder_seps_filename[] = TLOG_FILENAME_T(TLOG_FOLDER_SEPS);

// Return utc offset in minutes or throw tlog_ex on failure
//TURBO_DLL int utc_minutes_offset(const std::tm &tm = details::os::localtime());

TURBO_DLL std::string filename_to_str(const filename_t &filename);

#if (defined(TLOG_WCHAR_TO_UTF8_SUPPORT) || defined(TLOG_WCHAR_FILENAMES)) && defined(_WIN32)
TURBO_DLL void wstr_to_utf8buf(wstring_view_t wstr, memory_buf_t &target);

TURBO_DLL void utf8_to_wstrbuf(std::string_view str, wmemory_buf_t &target);
#endif


} // namespace os
} // namespace details
} // namespace turbo::tlog
