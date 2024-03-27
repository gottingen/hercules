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
#include <collie/log/details/os.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <tuple>

namespace clog {
namespace details {

inline file_helper::file_helper(const file_event_handlers &event_handlers)
    : event_handlers_(event_handlers) {}

inline file_helper::~file_helper() { close(); }

inline void file_helper::open(const filename_t &fname, bool truncate) {
    close();
    filename_ = fname;

    auto *mode = CLOG_FILENAME_T("ab");
    auto *trunc_mode = CLOG_FILENAME_T("wb");

    if (event_handlers_.before_open) {
        event_handlers_.before_open(filename_);
    }
    for (int tries = 0; tries < open_tries_; ++tries) {
        // create containing folder if not exists already.
        os::create_dir(os::dir_name(fname));
        if (truncate) {
            // Truncate by opening-and-closing a tmp file in "wb" mode, always
            // opening the actual log-we-write-to in "ab" mode, since that
            // interacts more politely with eternal processes that might
            // rotate/truncate the file underneath us.
            std::FILE *tmp;
            if (os::fopen_s(&tmp, fname, trunc_mode)) {
                continue;
            }
            std::fclose(tmp);
        }
        if (!os::fopen_s(&fd_, fname, mode)) {
            if (event_handlers_.after_open) {
                event_handlers_.after_open(filename_, fd_);
            }
            return;
        }

        details::os::sleep_for_millis(open_interval_);
    }

    throw_clog_ex("Failed opening file " + os::filename_to_str(filename_) + " for writing",
                    errno);
}

inline void file_helper::reopen(bool truncate) {
    if (filename_.empty()) {
        throw_clog_ex("Failed re opening file - was not opened before");
    }
    this->open(filename_, truncate);
}

inline void file_helper::flush() {
    if (std::fflush(fd_) != 0) {
        throw_clog_ex("Failed flush to file " + os::filename_to_str(filename_), errno);
    }
}

inline void file_helper::sync() {
    if (!os::fsync(fd_)) {
        throw_clog_ex("Failed to fsync file " + os::filename_to_str(filename_), errno);
    }
}

inline void file_helper::close() {
    if (fd_ != nullptr) {
        if (event_handlers_.before_close) {
            event_handlers_.before_close(filename_, fd_);
        }

        std::fclose(fd_);
        fd_ = nullptr;

        if (event_handlers_.after_close) {
            event_handlers_.after_close(filename_);
        }
    }
}

inline void file_helper::write(const memory_buf_t &buf) {
    if(fd_ == nullptr) return;
    size_t msg_size = buf.size();
    auto data = buf.data();
    if (std::fwrite(data, 1, msg_size, fd_) != msg_size) {
        throw_clog_ex("Failed writing to file " + os::filename_to_str(filename_), errno);
    }
}

inline size_t file_helper::size() const {
    if (fd_ == nullptr) {
        throw_clog_ex("Cannot use size() on closed file " + os::filename_to_str(filename_));
    }
    return os::filesize(fd_);
}

inline const filename_t &file_helper::filename() const { return filename_; }

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
inline std::tuple<filename_t, filename_t> file_helper::split_by_extension(
    const filename_t &fname) {
    auto ext_index = fname.rfind('.');

    // no valid extension found - return whole path and empty string as
    // extension
    if (ext_index == filename_t::npos || ext_index == 0 || ext_index == fname.size() - 1) {
        return std::make_tuple(fname, filename_t());
    }

    // treat cases like "/etc/rc.d/somelogfile or "/abc/.hiddenfile"
    auto folder_index = fname.find_last_of(details::os::folder_seps_filename);
    if (folder_index != filename_t::npos && folder_index >= ext_index - 1) {
        return std::make_tuple(fname, filename_t());
    }

    // finally - return a valid base and extension tuple
    return std::make_tuple(fname.substr(0, ext_index), fname.substr(ext_index));
}

}  // namespace details
}  // namespace clog
