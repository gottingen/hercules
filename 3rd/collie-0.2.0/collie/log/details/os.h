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

#include <ctime>  // std::time_t
#include <collie/log/common.h>

namespace clog::details::os {

    clog::log_clock::time_point now() noexcept;

    std::tm localtime(const std::time_t &time_tt) noexcept;

    std::tm localtime() noexcept;

    std::tm gmtime(const std::time_t &time_tt) noexcept;

    std::tm gmtime() noexcept;

// eol definition
#if !defined(CLOG_EOL)
#ifdef _WIN32
#define CLOG_EOL "\r\n"
#else
#define CLOG_EOL "\n"
#endif
#endif

    constexpr static const char *default_eol = CLOG_EOL;

// folder separator
#if !defined(CLOG_FOLDER_SEPS)
#ifdef _WIN32
#define CLOG_FOLDER_SEPS "\\/"
#else
#define CLOG_FOLDER_SEPS "/"
#endif
#endif

    constexpr static const char folder_seps[] = CLOG_FOLDER_SEPS;
    constexpr static const filename_t::value_type folder_seps_filename[] =
            CLOG_FILENAME_T(CLOG_FOLDER_SEPS);

    // fopen_s on non windows for writing
    bool fopen_s(FILE **fp, const filename_t &filename, const filename_t &mode);

    // Remove filename. return 0 on success
    int remove(const filename_t &filename) noexcept;

    // Remove file if exists. return 0 on success
    // Note: Non atomic (might return failure to delete if concurrently deleted by other process/thread)
    int remove_if_exists(const filename_t &filename) noexcept;

    int rename(const filename_t &filename1, const filename_t &filename2) noexcept;

    // Return if file exists.
    bool path_exists(const filename_t &filename) noexcept;

    // Return file size according to open FILE* object
    size_t filesize(FILE *f);

    // Return utc offset in minutes or throw CLogEx on failure
    int utc_minutes_offset(const std::tm &tm = details::os::localtime());

    // Return current thread id as size_t
    // It exists because the std::this_thread::get_id() is much slower(especially
    // under VS 2013)
    size_t _thread_id() noexcept;

    // Return current thread id as size_t (from thread local storage)
    size_t thread_id() noexcept;

    // This is avoid msvc issue in sleep_for that happens if the clock changes.
    void sleep_for_millis(unsigned int milliseconds) noexcept;

    std::string filename_to_str(const filename_t &filename);

    int pid() noexcept;

    // Determine if the terminal supports colors
    // Source: https://github.com/agauniyal/rang/
    bool is_color_terminal() noexcept;

    // Determine if the terminal attached
    // Source: https://github.com/agauniyal/rang/
    bool in_terminal(FILE *file) noexcept;

#if (defined(CLOG_WCHAR_TO_UTF8_SUPPORT) || defined(CLOG_WCHAR_FILENAMES)) && defined(_WIN32)
    void wstr_to_utf8buf(wstring_view_t wstr, memory_buf_t &target);

    void utf8_to_wstrbuf(string_view_t str, wmemory_buf_t &target);
#endif

    // Return directory name from given path or empty string
    // "abc/file" => "abc"
    // "abc/" => "abc"
    // "abc" => ""
    // "abc///" => "abc//"
    filename_t dir_name(const filename_t &path);

    // Create a dir from the given path.
    // Return true if succeeded or if this dir already exists.
    bool create_dir(const filename_t &path);

    // non thread safe, cross platform getenv/getenv_s
    // return empty string if field not found
    std::string getenv(const char *field);

    // Do fsync by FILE objectpointer.
    // Return true on success.
    bool fsync(FILE *fp);

}  // namespace clog::details::os

#include <collie/log/details/os-inl.h>
