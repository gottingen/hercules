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
// Created by jeff on 23-11-28.
//

#include "turbo/files/sys/random_write_file.h"
#include "turbo/files/sys/sys_io.h"
#include "turbo/log/logging.h"

namespace turbo {

    RandomWriteFile::RandomWriteFile(const FileEventListener &listener) : _listener(listener) {

    }

    RandomWriteFile::~RandomWriteFile() {
        close();
    }


    turbo::Status
    RandomWriteFile::open(const turbo::filesystem::path &fname, const OpenOption &option) noexcept {
        close();
        _option = option;
        _file_path = fname;
        TURBO_ASSERT(!_file_path.empty());
        auto *mode = "ab";
        auto *trunc_mode = "wb";

        if (_listener.before_open) {
            _listener.before_open(_file_path);
        }
        for (int tries = 0; tries < _option.open_tries; ++tries) {
            // create containing folder if not exists already.
            if (_option.create_dir_if_miss) {
                auto pdir = _file_path.parent_path();
                if (!pdir.empty()) {
                    std::error_code ec;
                    if (!turbo::filesystem::exists(pdir, ec)) {
                        if (ec) {
                            continue;
                        }
                        if (!turbo::filesystem::create_directories(pdir, ec)) {
                            continue;
                        }
                    }
                }
            }
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
        return turbo::errno_to_status(errno, turbo::format("Failed opening file {} for writing", _file_path.c_str()));
    }

    turbo::Status RandomWriteFile::reopen(bool truncate) {
        close();
        if (_file_path.empty()) {
            return turbo::make_status(kEINVAL);
        }
        OpenOption option = _option;
        if (truncate) {
            option.truncate();
        }
        return open(_file_path, option);
    }

    turbo::Status RandomWriteFile::write(off_t offset, const void *data, size_t size, bool truncate) {
        if (_fd == -1) {
            return turbo::make_status(kEBADFD);
        }

        ssize_t write_size = ::pwrite(_fd, data, size, static_cast<off_t>(offset));
        if (write_size < 0) {
            return turbo::errno_to_status(errno, _file_path.c_str());
        }
        if (truncate) {
            if (::ftruncate(_fd, static_cast<off_t>(offset + size)) != 0) {
                return turbo::errno_to_status(errno,
                                              turbo::format("Failed truncate file {} for size:{} ", _file_path.c_str(),
                                                            static_cast<off_t>(offset + size)));
            }
        }
        return turbo::ok_status();
    }

    [[nodiscard]] turbo::Status
    RandomWriteFile::write(off_t offset, const turbo::IOBuf &buff, bool truncate) {
        size_t size = buff.size();
        IOBuf piece_data(buff);
        off_t orig_offset = offset;
        ssize_t left = size;
        while (left > 0) {
            auto wrs = piece_data.pcut_into_file_descriptor(_fd, offset, left);
            if (wrs.ok() && wrs.value() >= 0) {
                offset += wrs.value();
                left -= wrs.value();
            } else if (errno == EINTR) {
                continue;
            } else {
                TLOG_WARN("write falied, err: {} fd: {} offset: {} size: {}", wrs.status().to_string(), _fd,
                          orig_offset, size);
                return wrs.status();
            }
        }

        if (truncate) {
            if (::ftruncate(_fd, static_cast<off_t>(offset + size)) != 0) {
                return turbo::errno_to_status(errno,
                                              turbo::format("Failed truncate file {} for size:{} ", _file_path.c_str(),
                                                            static_cast<off_t>(offset + size)));
            }
        }

        return turbo::ok_status();
    }

    turbo::Status RandomWriteFile::truncate(size_t size) {
        if (::ftruncate(_fd, static_cast<off_t>(size)) != 0) {
            return turbo::errno_to_status(errno,
                                          turbo::format("Failed truncate file {} for size:{} ", _file_path.c_str(),
                                                        static_cast<off_t>(size)));
        }
        return turbo::ok_status();
    }

    turbo::ResultStatus<size_t> RandomWriteFile::size() const {
        INVALID_FD_RETURN(_fd);
        auto rs = turbo::file_size(_fd);
        if(rs == -1) {
            return turbo::make_status();
        }
        return rs;
    }

    void RandomWriteFile::close() {
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

    turbo::Status RandomWriteFile::flush() {
        INVALID_FD_RETURN(_fd);
        if (::fdatasync(_fd) != 0) {
            return turbo::errno_to_status(errno,
                                          turbo::format("Failed flush to file {}", _file_path.c_str()));
        }
        return turbo::ok_status();
    }

    const turbo::filesystem::path &RandomWriteFile::file_path() const {
        return _file_path;
    }

}  // namespace turbo