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

#include <turbo/log/sinks/ansicolor_sink.h>
#include "turbo/system/sysinfo.h"
#include <turbo/log/pattern_formatter.h>
#include "turbo/log/details/os.h"
#include "turbo/format/print.h"

namespace turbo::tlog {
    namespace sinks {

        template<typename ConsoleMutex>
        ansicolor_sink<ConsoleMutex>::ansicolor_sink(FILE *target_file, color_mode mode)
                : target_file_(target_file), mutex_(ConsoleMutex::mutex()),
                  formatter_(details::make_unique<turbo::tlog::pattern_formatter>()) {
            set_color_mode(mode);
            colors_[level::trace] = fg(color::white);
            colors_[level::debug] = fg(color::cyan);
            colors_[level::info] = fg(color::green_yellow);
            colors_[level::warn] = fg(color::yellow);
            colors_[level::err] = fg(color::red) | emphasis::bold;
            colors_[level::critical] = fg(color::white) | bg(color::red)| emphasis::bold;;
            colors_[level::off] = fg(color::white);
        }

        template<typename ConsoleMutex>
        void ansicolor_sink<ConsoleMutex>::set_color(level::level_enum color_level, text_style color) {
            std::lock_guard<mutex_t> lock(mutex_);
            colors_[static_cast<size_t>(color_level)] = color;
        }

        template<typename ConsoleMutex>
        void ansicolor_sink<ConsoleMutex>::log(const details::log_msg &msg) {
            // Wrap the originally formatted message in color codes.
            // If color is not supported in the terminal, log as is instead.
            std::lock_guard<mutex_t> lock(mutex_);
            msg.color_range_start = 0;
            msg.color_range_end = 0;
            memory_buf_t formatted;
            formatter_->format(msg, formatted);
            if (should_do_colors_ && msg.color_range_end > msg.color_range_start) {
                // before color range
                print_range_(formatted, 0, msg.color_range_start);
                // in color range
                //print_ccode_(colors_[static_cast<size_t>(msg.level)]);
                //print_range_(formatted, msg.color_range_start, msg.color_range_end);
                //print_ccode_(reset);
                turbo::print(target_file_, colors_[static_cast<size_t>(msg.level)], std::string_view {formatted.data() + msg.color_range_start, msg.color_range_end - msg.color_range_start});
                // after color range
                print_range_(formatted, msg.color_range_end, formatted.size());
            } else {
                print_range_(formatted, 0, formatted.size());
            }
            fflush(target_file_);
        }

        template<typename ConsoleMutex>
        void ansicolor_sink<ConsoleMutex>::flush() {
            std::lock_guard<mutex_t> lock(mutex_);
            fflush(target_file_);
        }

        template<typename ConsoleMutex>
        void ansicolor_sink<ConsoleMutex>::set_pattern(const std::string &pattern) {
            std::lock_guard<mutex_t> lock(mutex_);
            formatter_ = std::unique_ptr<turbo::tlog::formatter>(new pattern_formatter(pattern));
        }

        template<typename ConsoleMutex>
        void ansicolor_sink<ConsoleMutex>::set_formatter(std::unique_ptr<turbo::tlog::formatter> sink_formatter) {
            std::lock_guard<mutex_t> lock(mutex_);
            formatter_ = std::move(sink_formatter);
        }

        template<typename ConsoleMutex>
        bool ansicolor_sink<ConsoleMutex>::should_color() {
            return should_do_colors_;
        }

        template<typename ConsoleMutex>
        void ansicolor_sink<ConsoleMutex>::set_color_mode(color_mode mode) {
            switch (mode) {
                case color_mode::always:
                    should_do_colors_ = true;
                    return;
                case color_mode::automatic:
                    should_do_colors_ = turbo::in_terminal(target_file_) && turbo::is_color_terminal();
                    return;
                case color_mode::never:
                    should_do_colors_ = false;
                    return;
                default:
                    should_do_colors_ = false;
            }
        }

        template<typename ConsoleMutex>
        void ansicolor_sink<ConsoleMutex>::print_ccode_(const std::string_view &color_code) {
            fwrite(color_code.data(), sizeof(char), color_code.size(), target_file_);
        }

        template<typename ConsoleMutex>
        void ansicolor_sink<ConsoleMutex>::print_range_(const memory_buf_t &formatted, size_t start, size_t end) {
            fwrite(formatted.data() + start, sizeof(char), end - start, target_file_);
        }

        template<typename ConsoleMutex>
        std::string ansicolor_sink<ConsoleMutex>::to_string_(const std::string_view &sv) {
            return std::string(sv.data(), sv.size());
        }

        // ansicolor_stdout_sink
        template<typename ConsoleMutex>
        ansicolor_stdout_sink<ConsoleMutex>::ansicolor_stdout_sink(color_mode mode)
                : ansicolor_sink<ConsoleMutex>(stdout, mode) {}

        // ansicolor_stderr_sink
        template<typename ConsoleMutex>
        ansicolor_stderr_sink<ConsoleMutex>::ansicolor_stderr_sink(color_mode mode)
                : ansicolor_sink<ConsoleMutex>(stderr, mode) {}

    } // namespace sinks
} // namespace turbo::tlog
