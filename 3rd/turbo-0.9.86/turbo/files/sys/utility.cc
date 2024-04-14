// Copyright 2023 The Turbo Authors.
//
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
#include "turbo/files/sys/utility.h"
#include "turbo/files/sys/sequential_read_file.h"
#include "turbo/files/sys/sequential_write_file.h"
#include "turbo/crypto/md5.h"
#include "turbo/crypto/sha1.h"

namespace turbo::sys_io {

    turbo::Status list_files(const std::string_view &root_path,std::vector<std::string> &result, bool full_path) noexcept{
        std::error_code ec;
        turbo::filesystem::directory_iterator itr(root_path, ec);
        if(ec) {
            return turbo::make_status(ec.value(), "open directory error:{}", ec.message());
        }
        turbo::filesystem::directory_iterator end;
        for(;itr != end;++itr) {
            if(!itr->is_directory(ec)) {
                if(ec) {
                    return turbo::make_status(ec.value(), "test if file error:{}", ec.message());
                }
                if(full_path) {
                    result.emplace_back(itr->path().string());
                } else {
                    result.emplace_back(itr->path().filename());
                }
            }
        }
        return turbo::ok_status();
    }

    turbo::Status list_directories(const std::string_view &root_path,std::vector<std::string> &result, bool full_path) noexcept {
        std::error_code ec;
        turbo::filesystem::directory_iterator itr(root_path, ec);
        if(ec) {
            return turbo::make_status(ec.value(), "open directory error:{}", ec.message());
        }
        turbo::filesystem::directory_iterator end;
        for(;itr != end;++itr) {
            if(itr->is_directory(ec)) {
                if(ec) {
                    return turbo::make_status(ec.value(), "test if file error:{}", ec.message());
                }
                if(full_path) {
                    result.emplace_back(itr->path().string());
                } else {
                    result.emplace_back(itr->path().filename());
                }
            }
        }
        return turbo::ok_status();
    }

    turbo::Status read_file(const std::string_view &file_path, std::string &result, bool append) noexcept {
        if(!append) {
            result.clear();
        }
        SequentialReadFile file;
        auto rs = file.open(file_path);
        if(!rs.ok()) {
            return rs;
        }
        auto r = file.read(&result);
        if(!r.ok()) {
            return r.status();
        }
        file.close();
        return turbo::ok_status();
    }

    turbo::Status write_file(const std::string_view &file_path, const std::string_view &content, bool truncate) noexcept {
        SequentialWriteFile file;
        auto rs = file.open(file_path, truncate);
        if(!rs.ok()) {
            return rs;
        }

        rs = file.write(content);
        if(!rs.ok()) {
            return rs;
        }
        file.close();
        return turbo::ok_status();
    }
}  // namespace turbo::sys_io
