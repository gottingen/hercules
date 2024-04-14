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

#    include <turbo/log/details/os.h>

#include "turbo/log/common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <array>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32

#    include <io.h>      // _get_osfhandle and _isatty support
#    include <process.h> //  _get_pid support
#    include <turbo/log/details/windows_include.h>

#    ifdef __MINGW32__
#        include <share.h>
#    endif

#    if defined(TLOG_WCHAR_TO_UTF8_SUPPORT) || defined(TLOG_WCHAR_FILENAMES)
#        include <limits>
#    endif

#    include <direct.h> // for _mkdir/_wmkdir

#else // unix

#    include <fcntl.h>
#    include <unistd.h>

#    ifdef __linux__

#        include <sys/syscall.h> //Use gettid() syscall under linux to get thread id

#    elif defined(_AIX)
#        include <pthread.h> // for pthread_getthrds_np

#    elif defined(__DragonFly__) || defined(__FreeBSD__)
#        include <pthread_np.h> // for pthread_getthreadid_np

#    elif defined(__NetBSD__)
#        include <lwp.h> // for _lwp_self

#    elif defined(__sun)
#        include <thread.h> // for thr_self
#    endif

#endif // unix

#ifndef __has_feature          // Clang - feature checking macros.
#    define __has_feature(x) 0 // Compatibility with non-clang compilers.
#endif

namespace turbo::tlog {
    namespace details {
        namespace os {
            /*
            turbo::tlog::log_clock::time_point now() noexcept {

#if defined __linux__ && defined TLOG_CLOCK_COARSE
                timespec ts;
                ::clock_gettime(CLOCK_REALTIME_COARSE, &ts);
                return std::chrono::time_point<log_clock, typename log_clock::duration>(
                    std::chrono::duration_cast<typename log_clock::duration>(std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec)));

#else
                return log_clock::now();
#endif
            }*/

            std::tm localtime(const std::time_t &time_tt) noexcept {

#ifdef _WIN32
                std::tm tm;
                ::localtime_s(&tm, &time_tt);
#else
                std::tm tm;
                ::localtime_r(&time_tt, &tm);
#endif
                return tm;
            }

            std::tm localtime() noexcept {
                std::time_t now_t = ::time(nullptr);
                return localtime(now_t);
            }

            std::tm gmtime(const std::time_t &time_tt) noexcept {

#ifdef _WIN32
                std::tm tm;
                ::gmtime_s(&tm, &time_tt);
#else
                std::tm tm;
                ::gmtime_r(&time_tt, &tm);
#endif
                return tm;
            }

            std::tm gmtime() noexcept {
                std::time_t now_t = ::time(nullptr);
                return gmtime(now_t);
            }

// Return utc offset in minutes or throw tlog_ex on failure
            int utc_minutes_offset(const std::tm &tm) {

#ifdef _WIN32
#    if _WIN32_WINNT < _WIN32_WINNT_WS08
                TIME_ZONE_INFORMATION tzinfo;
                auto rv = ::GetTimeZoneInformation(&tzinfo);
#    else
                DYNAMIC_TIME_ZONE_INFORMATION tzinfo;
                auto rv = ::GetDynamicTimeZoneInformation(&tzinfo);
#    endif
                if (rv == TIME_ZONE_ID_INVALID)
                    throw_tlog_ex("Failed getting timezone info. ", errno);

                int offset = -tzinfo.Bias;
                if (tm.tm_isdst)
                {
                    offset -= tzinfo.DaylightBias;
                }
                else
                {
                    offset -= tzinfo.StandardBias;
                }
                return offset;
#else

#    if defined(sun) || defined(__sun) || defined(_AIX) || (!defined(_BSD_SOURCE) && !defined(_GNU_SOURCE))
                // 'tm_gmtoff' field is BSD extension and it's missing on SunOS/Solaris
                struct helper
                {
                    static long int calculate_gmt_offset(const std::tm &localtm = details::os::localtime(), const std::tm &gmtm = details::os::gmtime())
                    {
                        int local_year = localtm.tm_year + (1900 - 1);
                        int gmt_year = gmtm.tm_year + (1900 - 1);

                        long int days = (
                            // difference in day of year
                            localtm.tm_yday -
                            gmtm.tm_yday

                            // + intervening leap days
                            + ((local_year >> 2) - (gmt_year >> 2)) - (local_year / 100 - gmt_year / 100) +
                            ((local_year / 100 >> 2) - (gmt_year / 100 >> 2))

                            // + difference in years * 365 */
                            + static_cast<long int>(local_year - gmt_year) * 365);

                        long int hours = (24 * days) + (localtm.tm_hour - gmtm.tm_hour);
                        long int mins = (60 * hours) + (localtm.tm_min - gmtm.tm_min);
                        long int secs = (60 * mins) + (localtm.tm_sec - gmtm.tm_sec);

                        return secs;
                    }
                };

                auto offset_seconds = helper::calculate_gmt_offset(tm);
#    else
                auto offset_seconds = tm.tm_gmtoff;
#    endif

                return static_cast<int>(offset_seconds / 60);
#endif
            }


// wchar support for windows file names (TLOG_WCHAR_FILENAMES must be defined)
#if defined(_WIN32) && defined(TLOG_WCHAR_FILENAMES)
            std::string filename_to_str(const filename_t &filename)
           {
               memory_buf_t buf;
               wstr_to_utf8buf(filename, buf);
               return TLOG_BUF_TO_STRING(buf);
           }
#else

            std::string filename_to_str(const filename_t &filename) {
                return filename;
            }

#endif

#if (defined(TLOG_WCHAR_TO_UTF8_SUPPORT) || defined(TLOG_WCHAR_FILENAMES)) && defined(_WIN32)
            void wstr_to_utf8buf(wstring_view_t wstr, memory_buf_t &target)
           {
               if (wstr.size() > static_cast<size_t>((std::numeric_limits<int>::max)()) / 2 - 1)
               {
                   throw_tlog_ex("UTF-16 string is too big to be converted to UTF-8");
               }

               int wstr_size = static_cast<int>(wstr.size());
               if (wstr_size == 0)
               {
                   target.resize(0);
                   return;
               }

               int result_size = static_cast<int>(target.capacity());
               if ((wstr_size + 1) * 2 > result_size)
               {
                   result_size = ::WideCharToMultiByte(CP_UTF8, 0, wstr.data(), wstr_size, NULL, 0, NULL, NULL);
               }

               if (result_size > 0)
               {
                   target.resize(result_size);
                   result_size = ::WideCharToMultiByte(CP_UTF8, 0, wstr.data(), wstr_size, target.data(), result_size, NULL, NULL);

                   if (result_size > 0)
                   {
                       target.resize(result_size);
                       return;
                   }
               }

               throw_tlog_ex(turbo::format("WideCharToMultiByte failed. Last error: {}", ::GetLastError()));
           }

            void utf8_to_wstrbuf(std::string_view str, wmemory_buf_t &target)
           {
               if (str.size() > static_cast<size_t>((std::numeric_limits<int>::max)()) - 1)
               {
                   throw_tlog_ex("UTF-8 string is too big to be converted to UTF-16");
               }

               int str_size = static_cast<int>(str.size());
               if (str_size == 0)
               {
                   target.resize(0);
                   return;
               }

               int result_size = static_cast<int>(target.capacity());
               if (str_size + 1 > result_size)
               {
                   result_size = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str.data(), str_size, NULL, 0);
               }

               if (result_size > 0)
               {
                   target.resize(result_size);
                   result_size = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str.data(), str_size, target.data(), result_size);

                   if (result_size > 0)
                   {
                       target.resize(result_size);
                       return;
                   }
               }

               throw_tlog_ex(turbo::format("MultiByteToWideChar failed. Last error: {}", ::GetLastError()));
           }
#endif // (defined(TLOG_WCHAR_TO_UTF8_SUPPORT) || defined(TLOG_WCHAR_FILENAMES)) && defined(_WIN32)

// return true on success
            static bool mkdir_(const filename_t &path) {
#ifdef _WIN32
#    ifdef TLOG_WCHAR_FILENAMES
                return ::_wmkdir(path.c_str()) == 0;
#    else
                return ::_mkdir(path.c_str()) == 0;
#    endif
#else
                return ::mkdir(path.c_str(), mode_t(0755)) == 0;
#endif
            }

        } // namespace os
    } // namespace details
} // namespace turbo::tlog
