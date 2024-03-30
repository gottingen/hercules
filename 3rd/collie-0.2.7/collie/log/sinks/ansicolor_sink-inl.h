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

#include <collie/log/details/os.h>
#include <collie/log/pattern_formatter.h>

namespace clog {
namespace sinks {

template <typename ConsoleMutex>
inline ansicolor_sink<ConsoleMutex>::ansicolor_sink(FILE *target_file, color_mode mode)
    : target_file_(target_file),
      mutex_(ConsoleMutex::mutex()),
      formatter_(details::make_unique<clog::pattern_formatter>())

{
    set_color_mode(mode);
    colors_.at(level::trace) = to_string_(white);
    colors_.at(level::debug) = to_string_(cyan);
    colors_.at(level::info) = to_string_(green);
    colors_.at(level::warn) = to_string_(yellow_bold);
    colors_.at(level::err) = to_string_(red_bold);
    colors_.at(level::critical) = to_string_(bold_on_red);
    colors_.at(level::off) = to_string_(reset);
}

template <typename ConsoleMutex>
inline void ansicolor_sink<ConsoleMutex>::set_color(level::level_enum color_level,
                                                           string_view_t color) {
    std::lock_guard<mutex_t> lock(mutex_);
    colors_.at(static_cast<size_t>(color_level)) = to_string_(color);
}

template <typename ConsoleMutex>
inline void ansicolor_sink<ConsoleMutex>::log(const details::log_msg &msg) {
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
        print_ccode_(colors_.at(static_cast<size_t>(msg.level)));
        print_range_(formatted, msg.color_range_start, msg.color_range_end);
        print_ccode_(reset);
        // after color range
        print_range_(formatted, msg.color_range_end, formatted.size());
    } else  // no color
    {
        print_range_(formatted, 0, formatted.size());
    }
    fflush(target_file_);
}

template <typename ConsoleMutex>
inline void ansicolor_sink<ConsoleMutex>::flush() {
    std::lock_guard<mutex_t> lock(mutex_);
    fflush(target_file_);
}

template <typename ConsoleMutex>
inline void ansicolor_sink<ConsoleMutex>::set_pattern(const std::string &pattern) {
    std::lock_guard<mutex_t> lock(mutex_);
    formatter_ = std::unique_ptr<clog::formatter>(new pattern_formatter(pattern));
}

template <typename ConsoleMutex>
inline void ansicolor_sink<ConsoleMutex>::set_formatter(
    std::unique_ptr<clog::formatter> sink_formatter) {
    std::lock_guard<mutex_t> lock(mutex_);
    formatter_ = std::move(sink_formatter);
}

template <typename ConsoleMutex>
inline bool ansicolor_sink<ConsoleMutex>::should_color() {
    return should_do_colors_;
}

template <typename ConsoleMutex>
inline void ansicolor_sink<ConsoleMutex>::set_color_mode(color_mode mode) {
    switch (mode) {
        case color_mode::always:
            should_do_colors_ = true;
            return;
        case color_mode::automatic:
            should_do_colors_ =
                details::os::in_terminal(target_file_) && details::os::is_color_terminal();
            return;
        case color_mode::never:
            should_do_colors_ = false;
            return;
        default:
            should_do_colors_ = false;
    }
}

template <typename ConsoleMutex>
inline void ansicolor_sink<ConsoleMutex>::print_ccode_(const string_view_t &color_code) {
    fwrite(color_code.data(), sizeof(char), color_code.size(), target_file_);
}

template <typename ConsoleMutex>
inline void ansicolor_sink<ConsoleMutex>::print_range_(const memory_buf_t &formatted,
                                                              size_t start,
                                                              size_t end) {
    fwrite(formatted.data() + start, sizeof(char), end - start, target_file_);
}

template <typename ConsoleMutex>
inline std::string ansicolor_sink<ConsoleMutex>::to_string_(const string_view_t &sv) {
    return std::string(sv.data(), sv.size());
}

// ansicolor_stdout_sink
template <typename ConsoleMutex>
inline ansicolor_stdout_sink<ConsoleMutex>::ansicolor_stdout_sink(color_mode mode)
    : ansicolor_sink<ConsoleMutex>(stdout, mode) {}

// ansicolor_stderr_sink
template <typename ConsoleMutex>
inline ansicolor_stderr_sink<ConsoleMutex>::ansicolor_stderr_sink(color_mode mode)
    : ansicolor_sink<ConsoleMutex>(stderr, mode) {}

}  // namespace sinks
}  // namespace clog