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

#include "turbo/files/sys/sys_io.h"
#include "turbo/platform/port.h"
#if defined(TURBO_PLATFORM_POSIX)
namespace turbo::sys_io {
    turbo::ResultStatus<FILE_HANDLER> open_read(const turbo::filesystem::path &filename, const std::string &mode,
                                                const FileOption &option) {
        TURBO_UNUSED(mode);
        int flag = O_RDONLY;
        if (option.prevent_child) {
            flag |= O_CLOEXEC;
        }

        const FILE_HANDLER fd = ::open((filename.c_str()), flag, option.mode);
        if (fd == -1) {
            return turbo::errno_to_status(errno, "");
        }
        return fd;

    }

    turbo::ResultStatus<size_t> file_size(int fd) {
        if (fd == -1) {
            return turbo::make_status(kEINVAL);
        }
// 64 bits(but not in osx or cygwin, where fstat64 is deprecated)
#    if (defined(__linux__) || defined(__sun) || defined(_AIX)) && (defined(__LP64__) || defined(_LP64))
        struct stat64 st;
        if (::fstat64(fd, &st) == 0) {
            return static_cast<size_t>(st.st_size);
        }
#    else // other unix or linux 32 bits or cygwin
        struct stat st;
                if (::fstat(fd, &st) == 0)
                {
                    return static_cast<size_t>(st.st_size);
                }
#    endif
        return turbo::errno_to_status(errno, "Failed getting file size from fd");
    }

    turbo::ResultStatus<FILE_HANDLER>
    open_write(const turbo::filesystem::path &filename, const std::string &mode, const FileOption &option) {
        int mode_flag = (mode == "ab") ? O_APPEND : O_TRUNC;
        mode_flag |= O_CREAT;
        mode_flag |= O_WRONLY;
        if (option.prevent_child) {
            mode_flag |= O_CLOEXEC;
        }
        const int fd = ::open((filename.c_str()), mode_flag, option.mode);
        if (fd == -1) {
            return turbo::errno_to_status(errno, "");
        }
        return fd;
    }
}

#endif  // TURBO_PLATFORM_POSIX