// Copyright 2024 The EA Authors.
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
#include <turbo/system/process.h>

namespace turbo {

    Process::Process(const std::vector<string_type> &arguments, const string_type &path,
                     std::function<void(const char *bytes, size_t n)> read_stdout,
                     std::function<void(const char *bytes, size_t n)> read_stderr,
                     bool open_stdin, const Config &config) noexcept
            : closed(true), read_stdout(std::move(read_stdout)), read_stderr(std::move(read_stderr)),
              open_stdin(open_stdin), config(config) {
        open(arguments, path);
        async_read();
    }

    Process::Process(const string_type &command, const string_type &path,
                     std::function<void(const char *bytes, size_t n)> read_stdout,
                     std::function<void(const char *bytes, size_t n)> read_stderr,
                     bool open_stdin, const Config &config) noexcept
            : closed(true), read_stdout(std::move(read_stdout)), read_stderr(std::move(read_stderr)),
              open_stdin(open_stdin), config(config) {
        open(command, path);
        async_read();
    }

    Process::Process(const std::vector<string_type> &arguments, const string_type &path,
                     const environment_type &environment,
                     std::function<void(const char *bytes, size_t n)> read_stdout,
                     std::function<void(const char *bytes, size_t n)> read_stderr,
                     bool open_stdin, const Config &config) noexcept
            : closed(true), read_stdout(std::move(read_stdout)), read_stderr(std::move(read_stderr)),
              open_stdin(open_stdin), config(config) {
        open(arguments, path, &environment);
        async_read();
    }

    Process::Process(const string_type &command, const string_type &path,
                     const environment_type &environment,
                     std::function<void(const char *bytes, size_t n)> read_stdout,
                     std::function<void(const char *bytes, size_t n)> read_stderr,
                     bool open_stdin, const Config &config) noexcept
            : closed(true), read_stdout(std::move(read_stdout)), read_stderr(std::move(read_stderr)),
              open_stdin(open_stdin), config(config) {
        open(command, path, &environment);
        async_read();
    }

    Process::~Process() noexcept {
        close_fds();
    }

    Process::id_type Process::get_id() const noexcept {
        return data.id;
    }

    bool Process::write(const std::string &str) {
        return write(str.c_str(), str.size());
    }

} // namespace turbo
