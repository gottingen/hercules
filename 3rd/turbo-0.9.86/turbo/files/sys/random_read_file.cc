// Copyright 2022 The Turbo Authors.
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
#include "turbo/files/sys/random_read_file.h"
#include "turbo/base/casts.h"
#include "turbo/files/sys/sys_io.h"
#include "turbo/system/io.h"
#include "turbo/log/logging.h"
#include "turbo/times/clock.h"
#include "turbo/times/time.h"
#include "turbo/status/result_status.h"
#include "turbo/status/status.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <fcntl.h>
#include <unistd.h>
#include <utility>

namespace turbo {

    RandomReadFile::RandomReadFile(const FileEventListener &listener)
            : _listener(listener) {

    }

    RandomReadFile::~RandomReadFile() {
        this->close();
    }

    turbo::Status
    RandomReadFile::open(const turbo::filesystem::path &path, const turbo::OpenOption &option) noexcept {
        this->close();
        _option = option;
        _file_path = path;
        if(_file_path.empty()) {
            return turbo::make_status(kEINVAL, "file path is empty");
        }
        if (_listener.before_open) {
            _listener.before_open(_file_path);
        }

        for (int tries = 0; tries < _option.open_tries; ++tries) {
            auto rs = turbo::open_file(_file_path, _option);
            if (rs.ok()) {
                _fd = rs.value();
                if (_listener.after_open) {
                    _listener.after_open(_file_path, _fd);
                }
                return turbo::ok_status();
            }
            if (_option.open_interval > 0) {
                turbo::sleep_for(turbo::Duration::milliseconds(_option.open_interval));
            }
        }
        return turbo::make_status(errno, turbo::format("Failed opening file {} for reading", _file_path.c_str()));
    }

    turbo::ResultStatus<size_t> RandomReadFile::read(off_t offset, void *buff, size_t len) {
        INVALID_FD_RETURN(_fd);
        size_t has_read = 0;
        /// _fd may > 0 with _fp valid
        ssize_t read_size = sys_pread(_fd, buff, len, static_cast<off_t>(offset));
        if(read_size < 0 ) {
            return turbo::make_status();
        }
        // read_size > 0 means read the end of file
        return has_read;
    }

    turbo::ResultStatus<size_t> RandomReadFile::read(off_t offset, std::string *content, size_t n) {
        INVALID_FD_RETURN(_fd);
        size_t len = n;
        if(len == kInfiniteFileSize) {
            auto r = turbo::file_size(_fd);
            if(r == -1) {
                return turbo::make_status();
            }
            len = r - offset;
            if(len <= 0) {
                return turbo::make_status(kEINVAL, "bad offset");
            }
        }
        auto pre_len = content->size();
        content->resize(pre_len + len);
        char* pdata = content->data() + pre_len;
        auto rs = turbo::sys_pread(_fd, pdata, len, offset);
        if(rs < 0) {
            content->resize(pre_len);
            return make_status();
        }
        content->resize(pre_len + rs);
        return rs;
    }

    turbo::ResultStatus<size_t> RandomReadFile::read(off_t offset, turbo::IOBuf *buf, size_t n) {
        INVALID_FD_RETURN(_fd);
        size_t len = n;
        if(len == kInfiniteFileSize) {
            auto r = turbo::file_size(_fd);
            if(r == -1) {
                return turbo::make_status();
            }
            len = r - offset;
            if(len <= 0) {
                return turbo::make_status(kEINVAL, "bad offset");
            }
        }
        IOPortal portal;
        auto rs = portal.pappend_from_file_descriptor(_fd, offset, len);
        if(!rs.ok()) {
            return rs;
        }
        buf->append(std::move(portal));

        return rs.value();
    }

    void RandomReadFile::close() {
        if (_fd != INVALID_FILE_HANDLER) {
            if (_listener.before_close) {
                _listener.before_close(_file_path, _fd);
            }

            ::close(_fd);
            _fd = INVALID_FILE_HANDLER;

            if (_listener.after_close) {
                _listener.after_close(_file_path);
            }
        }
    }


} // namespace turbo