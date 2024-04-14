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
// Created by jeff on 24-1-13.
//

#ifndef TURBO_SYSTEM_IO_SYS_IO_H_
#define TURBO_SYSTEM_IO_SYS_IO_H_

#include "turbo/status/result_status.h"
#include "turbo/status/status.h"
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>

namespace turbo {

#if defined(TURBO_PLATFORM_LINUX)
    typedef int FILE_HANDLER;
    static constexpr FILE_HANDLER INVALID_FILE_HANDLER = -1;
#elif defined(TURBO_PLATFORM_WINDOWS)

#endif

    ssize_t sys_pwritev(FILE_HANDLER fd, const struct iovec *vector, int count, off_t offset);

    ssize_t sys_preadv(FILE_HANDLER fd, const struct iovec *vector, int count, off_t offset);


    ssize_t sys_pwrite(FILE_HANDLER fd, const void *data, int count, off_t offset);

    ssize_t sys_pread(FILE_HANDLER fd, const void *data, int count, off_t offset);

    ssize_t sys_writev(FILE_HANDLER fd, const struct iovec *vector, int count);

    ssize_t sys_readv(FILE_HANDLER fd, const struct iovec *vector, int count);

    ssize_t sys_write(FILE_HANDLER fd, const void *data, int count);

    ssize_t sys_read(FILE_HANDLER fd, const void *data, int count);

    ssize_t file_size(int fd);


    struct OpenOption {
        int32_t open_tries{1};
        uint32_t open_interval{0};
        int flags{0};
        int mode{0644};
        bool create_dir_if_miss{false};

        constexpr OpenOption() = default;

        constexpr OpenOption(const OpenOption &) = default;

        constexpr OpenOption(OpenOption &&) = default;

        constexpr OpenOption &operator=(const OpenOption &) = default;

        constexpr OpenOption &operator=(OpenOption &&) = default;

        constexpr OpenOption &tries(int32_t tries) {
            open_tries = tries;
            return *this;
        }

        constexpr OpenOption &interval(uint32_t interval) {
            open_interval = interval;
            return *this;
        }

        constexpr OpenOption &read_only() {
            flags |= O_RDONLY;
            return *this;
        }

        constexpr OpenOption &write_only() {
            flags |= O_WRONLY;
            return *this;
        }

        constexpr OpenOption &read_write() {
            flags |= O_RDWR;
            return *this;
        }

        constexpr OpenOption &append(bool append = true) {
            append ? flags |= O_APPEND : flags &= ~O_APPEND;
            return *this;
        }

        constexpr OpenOption &truncate(bool truncate = true) {
            truncate ? flags |= O_TRUNC : flags &= ~O_TRUNC;
            return *this;
        }

        constexpr OpenOption &create(bool create = true) {
            create ? flags |= O_CREAT : flags &= ~O_CREAT;
            return *this;
        }

        constexpr OpenOption &cloexec(bool cloexec = true) {
            cloexec ? flags |= O_CLOEXEC : flags &= ~O_CLOEXEC;
            return *this;
        }

        constexpr OpenOption &flag(int flag) {
            this->flags = flag;
            return *this;
        }

        constexpr OpenOption &create_dir(bool create_dir = true) {
            this->create_dir_if_miss = create_dir;
            return *this;
        }
    };

    static constexpr OpenOption kDefaultReadOption = OpenOption{1, 0, O_RDONLY | O_CLOEXEC, 0644, false};
    static constexpr OpenOption kDefaultAppendWriteOption = OpenOption{1, 0, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
                                                                       0644, false};
    static constexpr OpenOption kDefaultTruncateWriteOption = OpenOption{1, 0, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                                                                         0644, false};

    ResultStatus<FILE_HANDLER> open_file(const std::string &filename, const OpenOption &option);

    ResultStatus<FILE_HANDLER> open_file_read(const std::string &filename);

    ResultStatus<FILE_HANDLER> open_file_write(const std::string &filename, bool truncate = false);
}  // namespace turbo

#endif  // TURBO_SYSTEM_IO_SYS_IO_H_
