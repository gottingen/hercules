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
// Created by jeff on 24-1-9.
//

#ifndef TURBO_FILES_INTERNAL_SYS_ADAPTER_H_
#define TURBO_FILES_INTERNAL_SYS_ADAPTER_H_

#include "turbo/files/fwd.h"
#include "turbo/files/sys/sequential_read_file.h"
#include "turbo/files/sys/random_read_file.h"
#include "turbo/files/sys/sequential_write_file.h"
#include "turbo/files/sys/random_write_file.h"
#include "turbo/files/sys/temp_file.h"
#include "turbo/files/sys/utility.h"

namespace turbo {

    struct sys_adapter{};

    template<>
    struct FileAdapter<sys_adapter> {
        static SequentialFileReader* create_sequential_file_reader();

        static RandomAccessFileReader* create_random_file_reader();

        static SequentialFileWriter* create_sequential_file_writer();

        static RandomFileWriter* create_random_file_writer();

        static TempFileWriter* create_temp_file();

        static turbo::Status read_file(const std::string_view &file_path, std::string &result, bool append = false) noexcept;

        static turbo::Status write_file(const std::string_view &file_path, const std::string_view &content, bool truncate = true) noexcept;

        static turbo::Status list_files(const std::string_view &root_path,std::vector<std::string> &result, bool full_path = true) noexcept;

        static turbo::Status list_directories(const std::string_view &root_path,std::vector<std::string> &result, bool full_path = true) noexcept;

        static turbo::ResultStatus<size_t> file_size(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::ResultStatus<bool> exists(const std::string_view &dir_path) noexcept;

        static turbo::Status remove(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::Status remove_all(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::Status rename(const turbo::filesystem::path &old_path, const turbo::filesystem::path &new_path) noexcept;

        static turbo::Status create_directory(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::Status create_directories(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::Status resize_file(const turbo::filesystem::path &dir_path, size_t size) noexcept;

    };

    inline SequentialFileReader* FileAdapter<sys_adapter>::create_sequential_file_reader() {
        return new SequentialReadFile();
    }

    inline RandomAccessFileReader* FileAdapter<sys_adapter>::create_random_file_reader() {
        return new RandomReadFile();
    }

    inline SequentialFileWriter* FileAdapter<sys_adapter>::create_sequential_file_writer() {
        return new SequentialWriteFile();
    }

    inline RandomFileWriter* FileAdapter<sys_adapter>::create_random_file_writer() {
        return new RandomWriteFile();
    }

    inline TempFileWriter* FileAdapter<sys_adapter>::create_temp_file() {
        return new TempFile();
    }

    inline turbo::Status  FileAdapter<sys_adapter>::read_file(const std::string_view &file_path, std::string &result, bool append) noexcept {
        return sys_io::read_file(file_path, result, append);
    }

    inline turbo::Status FileAdapter<sys_adapter>::write_file(const std::string_view &file_path, const std::string_view &content, bool truncate) noexcept {
        return sys_io::write_file(file_path, content, truncate);
    }

    inline turbo::Status FileAdapter<sys_adapter>::list_files(const std::string_view &root_path,std::vector<std::string> &result, bool full_path) noexcept {
        return sys_io::list_files(root_path, result, full_path);
    }

    inline turbo::Status FileAdapter<sys_adapter>::list_directories(const std::string_view &root_path,std::vector<std::string> &result, bool full_path) noexcept {
        return sys_io::list_directories(root_path, result, full_path);
    }

    inline turbo::ResultStatus<size_t> FileAdapter<sys_adapter>::file_size(const turbo::filesystem::path &dir_path) noexcept {
        std::error_code ec;
        auto size = turbo::filesystem::file_size(dir_path, ec);
        if(ec){
            return turbo::errno_to_status(ec.value(), ec.message());
        }
        return size;
    }

    inline turbo::ResultStatus<bool> FileAdapter<sys_adapter>::exists(const std::string_view &dir_path) noexcept {
        std::error_code ec;
        auto exists = turbo::filesystem::exists(dir_path, ec);
        if(ec){
            return turbo::errno_to_status(ec.value(), ec.message());
        }
        return exists;
    }

    inline turbo::Status FileAdapter<sys_adapter>::remove(const turbo::filesystem::path &dir_path) noexcept {
        std::error_code ec;
        turbo::filesystem::remove(dir_path, ec);
        if(ec){
            return turbo::errno_to_status(ec.value(), ec.message());
        }
        return turbo::ok_status();
    }

    inline turbo::Status FileAdapter<sys_adapter>::remove_all(const turbo::filesystem::path &dir_path) noexcept {
        std::error_code ec;
        turbo::filesystem::remove_all(dir_path, ec);
        if(ec){
            return turbo::errno_to_status(ec.value(), ec.message());
        }
        return turbo::ok_status();
    }

    inline turbo::Status FileAdapter<sys_adapter>::rename(const turbo::filesystem::path &old_path, const turbo::filesystem::path &new_path) noexcept {
        std::error_code ec;
        turbo::filesystem::rename(old_path, new_path, ec);
        if(ec){
            return turbo::errno_to_status(ec.value(), ec.message());
        }
        return turbo::ok_status();
    }

    inline turbo::Status FileAdapter<sys_adapter>::create_directory(const turbo::filesystem::path &dir_path) noexcept {
        std::error_code ec;
        turbo::filesystem::create_directory(dir_path, ec);
        if(ec){
            return turbo::errno_to_status(ec.value(), ec.message());
        }
        return turbo::ok_status();
    }

    inline turbo::Status FileAdapter<sys_adapter>::create_directories(const turbo::filesystem::path &dir_path) noexcept {
        std::error_code ec;
        turbo::filesystem::create_directories(dir_path, ec);
        if(ec){
            return turbo::errno_to_status(ec.value(), ec.message());
        }
        return turbo::ok_status();
    }

    inline turbo::Status FileAdapter<sys_adapter>::resize_file(const turbo::filesystem::path &dir_path, size_t size) noexcept {
        std::error_code ec;
        turbo::filesystem::resize_file(dir_path, size, ec);
        if(ec){
            return turbo::errno_to_status(ec.value(), ec.message());
        }
        return turbo::ok_status();
    }
}  // namespace turbo
#endif  // TURBO_FILES_INTERNAL_SYS_ADAPTER_H_
