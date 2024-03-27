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

#ifdef _WIN32
#include <collie/log/sinks/wincolor_sink.h>
#else

#include <collie/log/sinks/ansicolor_sink.h>

#endif

#include <collie/log/details/synchronous_factory.h>

namespace clog {
    namespace sinks {
#ifdef _WIN32
        using stdout_color_sink_mt = wincolor_stdout_sink_mt;
        using stdout_color_sink_st = wincolor_stdout_sink_st;
        using stderr_color_sink_mt = wincolor_stderr_sink_mt;
        using stderr_color_sink_st = wincolor_stderr_sink_st;
#else
        using stdout_color_sink_mt = ansicolor_stdout_sink_mt;
        using stdout_color_sink_st = ansicolor_stdout_sink_st;
        using stderr_color_sink_mt = ansicolor_stderr_sink_mt;
        using stderr_color_sink_st = ansicolor_stderr_sink_st;
#endif
    }  // namespace sinks

    template<typename Factory = clog::synchronous_factory>
    std::shared_ptr<logger> stdout_color_mt(const std::string &logger_name,
                                            color_mode mode = color_mode::automatic);

    template<typename Factory = clog::synchronous_factory>
    std::shared_ptr<logger> stdout_color_st(const std::string &logger_name,
                                            color_mode mode = color_mode::automatic);

    template<typename Factory = clog::synchronous_factory>
    std::shared_ptr<logger> stderr_color_mt(const std::string &logger_name,
                                            color_mode mode = color_mode::automatic);

    template<typename Factory = clog::synchronous_factory>
    std::shared_ptr<logger> stderr_color_st(const std::string &logger_name,
                                            color_mode mode = color_mode::automatic);

}  // namespace clog

#include <collie/log/sinks/stdout_color_sinks-inl.h>
