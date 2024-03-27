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

#include <collie/log/common.h>
#include <tuple>

namespace clog {
    namespace details {

// Helper class for file sinks.
// When failing to open a file, retry several times(5) with a delay interval(10 ms).
// Throw CLogEx exception on errors.

        class file_helper {
        public:
            file_helper() = default;

            explicit file_helper(const file_event_handlers &event_handlers);

            file_helper(const file_helper &) = delete;

            file_helper &operator=(const file_helper &) = delete;

            ~file_helper();

            void open(const filename_t &fname, bool truncate = false);

            void reopen(bool truncate);

            void flush();

            void sync();

            void close();

            void write(const memory_buf_t &buf);

            size_t size() const;

            const filename_t &filename() const;

            //
            // return file path and its extension:
            //
            // "mylog.txt" => ("mylog", ".txt")
            // "mylog" => ("mylog", "")
            // "mylog." => ("mylog.", "")
            // "/dir1/dir2/mylog.txt" => ("/dir1/dir2/mylog", ".txt")
            //
            // the starting dot in filenames is ignored (hidden files):
            //
            // ".mylog" => (".mylog". "")
            // "my_folder/.mylog" => ("my_folder/.mylog", "")
            // "my_folder/.mylog.txt" => ("my_folder/.mylog", ".txt")
            static std::tuple<filename_t, filename_t> split_by_extension(const filename_t &fname);

        private:
            const int open_tries_ = 5;
            const unsigned int open_interval_ = 10;
            std::FILE *fd_{nullptr};
            filename_t filename_;
            file_event_handlers event_handlers_;
        };
    }  // namespace details
}  // namespace clog

#include <collie/log/details/file_helper-inl.h>
