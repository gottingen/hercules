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
#include "turbo/files/filesystem.h"
#include "turbo/log/details/null_mutex.h"
#include <turbo/format/format.h>
#include <turbo/format/fmt/chrono.h>
#include <turbo/log/sinks/base_sink.h>
#include "turbo/log/details/os.h"
#include <turbo/log/details/circular_q.h>
#include "turbo/log/details/synchronous_factory.h"
#include "turbo/strings/str_split.h"
#include <iostream>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace turbo::tlog {
    namespace sinks {

        /*
         * Generator of daily log file names in format basename.YYYY-MM-DD.ext
         */
        struct daily_filename_calculator {
            // Create filename for the form basename.YYYY-MM-DD
            static filename_t calc_filename(const filename_t &filename, const tm &now_tm) {
                filename_t basename, ext;
                std::tie(basename, ext) = turbo::split_by_extension(filename);
                return turbo::format(FMT_STRING(TLOG_FILENAME_T("{}_{:04d}-{:02d}-{:02d}{}")), basename,
                                       now_tm.tm_year + 1900,
                                       now_tm.tm_mon + 1, now_tm.tm_mday, ext);
            }
        };

        /*
         * Generator of daily log file names with strftime format.
         * Usages:
         *    auto sink =  std::make_shared<turbo::tlog::sinks::daily_file_format_sink_mt>("myapp-%Y-%m-%d:%H:%M:%S.log", hour, minute);"
         *    auto logger = turbo::tlog::daily_logger_format_mt("loggername, "myapp-%Y-%m-%d:%X.log", hour,  minute)"
         *
         */
        struct daily_filename_format_calculator {
            static filename_t calc_filename(const filename_t &filename, const tm &now_tm) {
                // generate fmt datetime format string, e.g. {:%Y-%m-%d}.
                filename_t fmt_filename = turbo::format(FMT_STRING(TLOG_FILENAME_T("{{:{}}}")),filename);
                // MSVC doesn't allow turbo::runtime(..) with wchar, with fmtlib versions < 9.1.x
                return turbo::format(turbo::runtime(fmt_filename), now_tm);

            }

        private:
#if defined __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif

            static size_t strftime(char *str, size_t count, const char *format, const std::tm *time) {
                return std::strftime(str, count, format, time);
            }

            static size_t strftime(wchar_t *str, size_t count, const wchar_t *format, const std::tm *time) {
                return std::wcsftime(str, count, format, time);
            }

#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
        };

        /*
         * Rotating file sink based on date.
         * If truncate != false , the created file will be truncated.
         * If max_files > 0, retain only the last max_files and delete previous.
         */
        template<typename Mutex, typename FileNameCalc = daily_filename_calculator>
        class daily_file_sink final : public base_sink<Mutex> {
        public:
            // create daily file sink which rotates on given time
            daily_file_sink(filename_t base_filename, int rotation_hour, int rotation_minute, bool truncate = false,
                            uint16_t max_files = 0,
                            const turbo::FileEventListener &event_handlers = {})
                    : base_filename_(std::move(base_filename)), rotation_h_(rotation_hour),
                      rotation_m_(rotation_minute), file_writer_{event_handlers}, truncate_(truncate),
                      max_files_(max_files), filenames_q_() {
                if (rotation_hour < 0 || rotation_hour > 23 || rotation_minute < 0 || rotation_minute > 59) {
                    throw_tlog_ex("daily_file_sink: Invalid rotation time in ctor");
                }

                auto now = turbo::Time::time_now();
                auto filename = FileNameCalc::calc_filename(base_filename_, now.to_local_tm());
                auto & open_option = truncate ? kLogTruncateOpenOption : kLogAppendOpenOption;
                auto r = file_writer_.open(filename, open_option);
                if(!r.ok()) {
                    throw_tlog_ex(r.to_string());
                }
                rotation_tp_ = next_rotation_tp_();

                if (max_files_ > 0) {
                    init_filenames_q_();
                }
            }

            filename_t filename() {
                std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
                return file_writer_.file_path().native();
            }

        protected:
            void sink_it_(const details::log_msg &msg) override {
                auto time = msg.time;
                bool should_rotate = time >= rotation_tp_;
                if (should_rotate) {
                    auto filename = FileNameCalc::calc_filename(base_filename_, time.to_local_tm());
                    auto & operation = truncate_ ? kLogTruncateOpenOption : kLogAppendOpenOption;
                    auto r = file_writer_.open(filename, operation);
                    if(!r.ok()) {
                        throw_tlog_ex(r.to_string());
                    }
                    rotation_tp_ = next_rotation_tp_();
                }
                memory_buf_t formatted;
                base_sink<Mutex>::formatter_->format(msg, formatted);
                auto r = file_writer_.write(formatted);
                if(!r.ok()) {
                    throw_tlog_ex(r.to_string());
                }

                // Do the cleaning only at the end because it might throw on failure.
                if (should_rotate && max_files_ > 0) {
                    delete_old_();
                }
            }

            void flush_() override {
                auto  r = file_writer_.flush();
                if(!r.ok()) {
                    throw_tlog_ex(r.to_string());
                }
            }

        private:
            void init_filenames_q_() {
                filenames_q_ = details::circular_q<filename_t>(static_cast<size_t>(max_files_));
                std::vector<filename_t> filenames;
                auto now = turbo::Time::time_now();
                while (filenames.size() < max_files_) {
                    auto filename = FileNameCalc::calc_filename(base_filename_, now.to_local_tm());
                    if (!turbo::filesystem::exists(filename)) {
                        break;
                    }
                    filenames.emplace_back(filename);
                    now -= turbo::Duration::hours(24);
                }
                for (auto iter = filenames.rbegin(); iter != filenames.rend(); ++iter) {
                    filenames_q_.push_back(std::move(*iter));
                }
            }

            turbo::Time next_rotation_tp_() {
                auto now = turbo::Time::time_now();
                auto date = now.to_local_tm();
                date.tm_hour = rotation_h_;
                date.tm_min = rotation_m_;
                date.tm_sec = 0;
                auto rotation_time = turbo::Time::from_tm(date, turbo::local_time_zone());
                if (rotation_time > now) {
                    return rotation_time;
                }
                return rotation_time + turbo::Duration::hours(24);
            }

            // Delete the file N rotations ago.
            // Throw tlog_ex on failure to delete the old file.
            void delete_old_() {
                using details::os::filename_to_str;

                filename_t current_file = file_writer_.file_path().native();
                if (filenames_q_.full()) {
                    auto old_filename = std::move(filenames_q_.front());
                    filenames_q_.pop_front();
                    std::error_code ec;
                    bool ok = true;
                    if(turbo::filesystem::exists(old_filename, ec)) {
                        turbo::filesystem::remove(old_filename, ec);
                        if(ec) {
                            ok = false;
                        }
                    }
                    if (!ok) {
                        filenames_q_.push_back(std::move(current_file));
                        throw_tlog_ex("Failed removing daily file " + filename_to_str(old_filename), errno);
                    }
                }
                filenames_q_.push_back(std::move(current_file));
            }

            filename_t base_filename_;
            int rotation_h_;
            int rotation_m_;
            turbo::Time rotation_tp_;
            turbo::SequentialWriteFile file_writer_;
            bool truncate_;
            uint16_t max_files_;
            details::circular_q<filename_t> filenames_q_;
        };

        using daily_file_sink_mt = daily_file_sink<std::mutex>;
        using daily_file_sink_st = daily_file_sink<details::null_mutex>;
        using daily_file_format_sink_mt = daily_file_sink<std::mutex, daily_filename_format_calculator>;
        using daily_file_format_sink_st = daily_file_sink<details::null_mutex, daily_filename_format_calculator>;

    } // namespace sinks

    //
    // factory functions
    //
    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    daily_logger_mt(const std::string &logger_name, const filename_t &filename, int hour = 0, int minute = 0,
                    bool truncate = false, uint16_t max_files = 0, const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::daily_file_sink_mt>(logger_name, filename, hour, minute, truncate,
                                                                   max_files, event_handlers);
    }

    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    daily_logger_format_mt(const std::string &logger_name, const filename_t &filename, int hour = 0,
                           int minute = 0, bool truncate = false, uint16_t max_files = 0,
                           const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::daily_file_format_sink_mt>(
                logger_name, filename, hour, minute, truncate, max_files, event_handlers);
    }

    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    daily_logger_st(const std::string &logger_name, const filename_t &filename, int hour = 0, int minute = 0,
                    bool truncate = false, uint16_t max_files = 0, const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::daily_file_sink_st>(logger_name, filename, hour, minute, truncate,
                                                                   max_files, event_handlers);
    }

    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    daily_logger_format_st(const std::string &logger_name, const filename_t &filename, int hour = 0,
                           int minute = 0, bool truncate = false, uint16_t max_files = 0,
                           const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::daily_file_format_sink_st>(
                logger_name, filename, hour, minute, truncate, max_files, event_handlers);
    }
} // namespace turbo::tlog
