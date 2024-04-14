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

#include "turbo/files/file_enumerator.h"

namespace turbo {

    turbo::Status FileEnumerator::open(const std::string &path) {
        FileEnumeratorOption option;
        return open(path, option);
    }

    turbo::Status FileEnumerator::open(const std::string &path, bool include_dirs, bool recursive) {
        FileEnumeratorOption option;
        option.include_dirs = include_dirs;
        option.recursive = recursive;
        return open(path, option);
    }

    turbo::Status FileEnumerator::open(const std::string &path, const FileEnumeratorOption &option) {
        option_ = option;
        std::error_code ec;
        if (!filesystem::exists(path_, ec)) {
            return turbo::make_status(kNotFound, "path not found");
        }
        if (!filesystem::is_directory(path_, ec)) {
            return turbo::make_status(kInvalidArgument, "path is not a directory");
        }

        path_ = path;
        rewind();
        return turbo::ok_status();
    }

    void FileEnumerator::rewind() {
        filesystem::directory_options dir_option = filesystem::directory_options::none;
        if (option_.follow_symlinks) {
            dir_option |= filesystem::directory_options::follow_directory_symlink;
        }
        if (option_.skip_permission_denied) {
            dir_option |= filesystem::directory_options::skip_permission_denied;
        }

        cache_.clear();
        cache_.reserve(option_.max_cache_size);
        if (option_.recursive) {
            riter_.reset(new filesystem::recursive_directory_iterator(path_, dir_option));
        } else {
            iter_.reset(new filesystem::directory_iterator(path_,dir_option));
        }
        fill_cache();
    }

    void FileEnumerator::fill_cache() {
        if (option_.recursive) {
            while (riter_->operator!=(filesystem::recursive_directory_iterator()) &&
                   cache_.size() < option_.max_cache_size && riter_->depth() < option_.max_depth) {
                if(!option_.include_dot_dot && ((*riter_)->path().filename() == ".." || (*riter_)->path().filename() == ".")){
                    ++(*riter_);
                    continue;
                }
                if(!option_.include_hidden && (*riter_)->path().filename().string().front() == '.'){
                    ++(*riter_);
                    continue;
                }
                auto &entry = *(*riter_);
                if (option_.filter && !option_.filter(entry)) {
                    continue;
                }
                if (option_.include_dirs && filesystem::is_directory(entry)) {
                    cache_.push_back(entry);
                } else if (option_.include_files && filesystem::is_regular_file(entry)) {
                    cache_.push_back(entry);
                }
                ++(*riter_);
            }
        } else {
            while (iter_->operator!=(filesystem::directory_iterator()) && cache_.size() < option_.max_cache_size) {
                auto &entry = *(*iter_);
                if(option_.include_dot_dot && ((*riter_)->path().filename() == ".." || (*riter_)->path().filename() == ".")){
                    ++(*riter_);
                    continue;
                }
                if(!option_.include_hidden && (*riter_)->path().filename().string().front() == '.'){
                    ++(*riter_);
                    continue;
                }
                if (option_.filter && !option_.filter(entry)) {
                    continue;
                }
                if (option_.include_dirs && filesystem::is_directory(entry)) {
                    cache_.push_back(entry);
                } else if (option_.include_files && filesystem::is_regular_file(entry)) {
                    cache_.push_back(entry);
                }
                ++(*iter_);
            }
        }
    }
    filesystem::directory_entry FileEnumerator::next() {
        if (cache_.empty()) {
            fill_cache();
        }
        if (cache_.empty()) {
            return filesystem::directory_entry();
        }
        auto entry = cache_.front();
        cache_.pop_front();
        return entry;
    }

    bool FileEnumerator::has_next() {
        if (cache_.empty()) {
            fill_cache();
        }
        return !cache_.empty();
    }
}  // namespace turbo