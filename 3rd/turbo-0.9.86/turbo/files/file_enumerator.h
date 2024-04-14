// Copyright 2023 The Elastic-AI Authors.
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
//
// Created by jeff on 24-1-26.
//

#ifndef TURBO_FILES_FILE_ENUMERATOR_H_
#define TURBO_FILES_FILE_ENUMERATOR_H_

#include "turbo/files/internal/filesystem.h"
#include "turbo/status/status.h"
#include <memory>
#include <vector>
#include <functional>
#include "turbo/container/ring_buffer.h"

namespace turbo {

    using FileEnumeratorFilter = std::function<bool(const filesystem::directory_entry &)>;

    struct FileEnumeratorOption {
        bool recursive{false};
        bool include_hidden{false};
        bool include_dirs{false};
        bool include_files{true};
        bool follow_symlinks{false};
        bool include_dot_dot{false};
        bool skip_permission_denied{true};
        FileEnumeratorFilter filter;
        size_t max_depth{std::numeric_limits<size_t>::max()};
        size_t max_cache_size{4096};
    };

    class FileEnumerator {
    public:
        FileEnumerator() = default;
        ~FileEnumerator() = default;

        FileEnumerator(const FileEnumerator &) = delete;
        FileEnumerator &operator=(const FileEnumerator &) = delete;

        turbo::Status open(const std::string &path);

        turbo::Status open(const std::string &path, bool include_dirs, bool recursive);

        turbo::Status open(const std::string &path, const FileEnumeratorOption &option);

        filesystem::directory_entry next();

        bool has_next();

        void rewind();
    private:
        void fill_cache();
        FileEnumeratorOption option_;
        turbo::filesystem::path path_;
        std::unique_ptr<turbo::filesystem::directory_iterator> iter_;
        std::unique_ptr<turbo::filesystem::recursive_directory_iterator> riter_;
        ring_buffer<filesystem::directory_entry> cache_;
    };
}  // namespace turbo
#endif // TURBO_FILES_FILE_ENUMERATOR_H_
