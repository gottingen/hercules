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
#include <turbo/log/sinks/base_sink.h>
#include "turbo/log/details/os.h"
#include <turbo/log/details/circular_q.h>
#include "turbo/log/details/synchronous_factory.h"
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace turbo::tlog {
    namespace sinks {

/*
 * Generator of Hourly log file names in format basename.YYYY-MM-DD-HH.ext
 */
        struct hourly_filename_calculator {
            // Create filename for the form basename.YYYY-MM-DD-H
            static filename_t calc_filename(const filename_t &filename, const tm &now_tm) {
                filename_t basename, ext;
                std::tie(basename, ext) = turbo::split_by_extension(filename);
                return turbo::format(TLOG_FILENAME_T("{}_{:04d}-{:02d}-{:02d}_{:02d}{}"), basename,
                                       now_tm.tm_year + 1900, now_tm.tm_mon + 1,
                                       now_tm.tm_mday, now_tm.tm_hour, ext);
            }
        };

/*
 * Rotating file sink based on time.
 * If truncate != false , the created file will be truncated.
 * If max_files > 0, retain only the last max_files and delete previous.
 */
        template<typename Mutex, typename FileNameCalc = hourly_filename_calculator>
        class hourly_file_sink final : public base_sink<Mutex> {
        public:
            // create hourly file sink which rotates on given time
            hourly_file_sink(
                    filename_t base_filename, bool truncate = false, uint16_t max_files = 0,
                    const turbo::FileEventListener &event_handlers = {})
                    : base_filename_(std::move(base_filename)), file_writer_{event_handlers}, truncate_(truncate),
                      max_files_(max_files), filenames_q_() {
                auto now = log_clock::now();
                auto filename = FileNameCalc::calc_filename(base_filename_, now_tm(now));
                auto r = file_writer_.open(filename, truncate_, kLogFileOption);
                if (!r.ok()) {
                    throw_tlog_ex(r.ToString());
                }
                auto rs = file_writer_.size();
                remove_init_file_ = (rs.ok() && rs.value() == 0);
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
                    if (remove_init_file_) {
                        file_writer_.close();
                        turbo::filesystem::remove(file_writer_.file_path());
                    }
                    auto filename = FileNameCalc::calc_filename(base_filename_, now_tm(time));
                    auto r = file_writer_.open(filename, truncate_);
                    if (!r.ok()) {
                        throw_tlog_ex(r.ToString());
                    }
                    rotation_tp_ = next_rotation_tp_();
                }
                remove_init_file_ = false;
                memory_buf_t formatted;
                base_sink<Mutex>::formatter_->format(msg, formatted);
                auto r = file_writer_.write(formatted);
                if (!r.ok()) {
                    throw_tlog_ex(r.to_string());
                }
                // Do the cleaning only at the end because it might throw on failure.
                if (should_rotate && max_files_ > 0) {
                    delete_old_();
                }
            }

            void flush_() override {
                auto r = file_writer_.flush();
                if (!r.ok()) {
                    throw_tlog_ex(r.to_string());
                }
            }

        private:
            void init_filenames_q_() {
                filenames_q_ = details::circular_q<filename_t>(static_cast<size_t>(max_files_));
                std::vector<filename_t> filenames;
                auto now = log_clock::now();
                while (filenames.size() < max_files_) {
                    auto filename = FileNameCalc::calc_filename(base_filename_, now_tm(now));
                    if (!path_exists(filename)) {
                        break;
                    }
                    filenames.emplace_back(filename);
                    now -= std::chrono::hours(1);
                }
                for (auto iter = filenames.rbegin(); iter != filenames.rend(); ++iter) {
                    filenames_q_.push_back(std::move(*iter));
                }
            }

            tm now_tm(log_clock::time_point tp) {
                time_t tnow = log_clock::to_time_t(tp);
                return turbo::tlog::details::os::localtime(tnow);
            }

            log_clock::time_point next_rotation_tp_() {
                auto now = log_clock::now();
                tm date = now_tm(now);
                date.tm_min = 0;
                date.tm_sec = 0;
                auto rotation_time = log_clock::from_time_t(std::mktime(&date));
                if (rotation_time > now) {
                    return rotation_time;
                }
                return {rotation_time + std::chrono::hours(1)};
            }

            // Delete the file N rotations ago.
            // Throw tlog_ex on failure to delete the old file.
            void delete_old_() {

                filename_t current_file = file_writer_.file_path().native();
                if (filenames_q_.full()) {
                    auto old_filename = std::move(filenames_q_.front());
                    filenames_q_.pop_front();
                    std::error_code ec;
                    if(turbo::filesystem::exists(old_filename, ec)) {
                        if(ec) {
                            filenames_q_.push_back(std::move(current_file));
                            throw_tlog_ex(ec.message());
                        }
                        auto r = turbo::filesystem::remove(old_filename, ec);
                        TURBO_UNUSED(r);
                        if (ec) {
                            filenames_q_.push_back(std::move(current_file));
                            throw_tlog_ex(ec.message());
                        }
                    }
                }
                filenames_q_.push_back(std::move(current_file));
            }

            filename_t base_filename_;
            log_clock::time_point rotation_tp_;
            turbo::SequentialWriteFile file_writer_;
            bool truncate_;
            uint16_t max_files_;
            details::circular_q<filename_t> filenames_q_;
            bool remove_init_file_;
        };

        using hourly_file_sink_mt = hourly_file_sink<std::mutex>;
        using hourly_file_sink_st = hourly_file_sink<details::null_mutex>;

    } // namespace sinks

//
// factory functions
//
    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    hourly_logger_mt(const std::string &logger_name, const filename_t &filename, bool truncate = false,
                     uint16_t max_files = 0, const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::hourly_file_sink_mt>(logger_name, filename, truncate, max_files,
                                                                    event_handlers);
    }

    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    hourly_logger_st(const std::string &logger_name, const filename_t &filename, bool truncate = false,
                     uint16_t max_files = 0, const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::hourly_file_sink_st>(logger_name, filename, truncate, max_files,
                                                                    event_handlers);
    }
} // namespace turbo::tlog
