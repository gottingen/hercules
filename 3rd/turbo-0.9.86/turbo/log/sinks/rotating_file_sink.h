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

#include "turbo/log/sinks/base_sink.h"
#include "turbo/files/filesystem.h"
#include "turbo/log/details/null_mutex.h"
#include "turbo/log/details/synchronous_factory.h"

#include <chrono>
#include <mutex>
#include <string>

namespace turbo::tlog {
    namespace sinks {

        //
        // Rotating file sink based on size
        //
        template<typename Mutex>
        class rotating_file_sink final : public base_sink<Mutex> {
        public:
            rotating_file_sink(filename_t base_filename, std::size_t max_size, std::size_t max_files,
                               bool rotate_on_open = false,
                               const turbo::FileEventListener &event_handlers = {});

            static filename_t calc_filename(const filename_t &filename, std::size_t index);

            filename_t filename();

        protected:
            void sink_it_(const details::log_msg &msg) override;

            void flush_() override;

        private:
            // Rotate files:
            // log.txt -> log.1.txt
            // log.1.txt -> log.2.txt
            // log.2.txt -> log.3.txt
            // log.3.txt -> delete
            void rotate_();

            // delete the target if exists, and rename the src file  to target
            // return true on success, false otherwise.
            bool rename_file_(const filename_t &src_filename, const filename_t &target_filename);

            filename_t base_filename_;
            std::size_t max_size_;
            std::size_t max_files_;
            std::size_t current_size_;
            turbo::SequentialWriteFile file_writer_;
        };

        using rotating_file_sink_mt = rotating_file_sink<std::mutex>;
        using rotating_file_sink_st = rotating_file_sink<details::null_mutex>;

    } // namespace sinks

    //
    // factory functions
    //

    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    rotating_logger_mt(const std::string &logger_name, const filename_t &filename, size_t max_file_size,
                       size_t max_files, bool rotate_on_open = false,
                       const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::rotating_file_sink_mt>(
                logger_name, filename, max_file_size, max_files, rotate_on_open, event_handlers);
    }

    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    rotating_logger_st(const std::string &logger_name, const filename_t &filename, size_t max_file_size,
                       size_t max_files, bool rotate_on_open = false,
                       const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::rotating_file_sink_st>(
                logger_name, filename, max_file_size, max_files, rotate_on_open, event_handlers);
    }
} // namespace turbo::tlog

