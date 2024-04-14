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
#include "turbo/system/io/sys_io.h"
#include "turbo/system/io/fd_guard.h"
#include "turbo/log/logging.h"
#include "turbo/status/error.h"  // errno
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace turbo::system_internal {
    typedef ssize_t (*iov_function)(int fd, const struct iovec *vector,
                                    int count, off_t offset);

    static ssize_t user_preadv(int fd, const struct iovec *vector,
                               int count, off_t offset) {
        ssize_t total_read = 0;
        for (int i = 0; i < count; ++i) {
            const ssize_t rc = ::pread(fd, vector[i].iov_base, vector[i].iov_len, offset);
            if (rc <= 0) {
                return total_read > 0 ? total_read : rc;
            }
            total_read += rc;
            offset += rc;
            if (rc < (ssize_t) vector[i].iov_len) {
                break;
            }
        }
        return total_read;
    }

    static ssize_t user_pwritev(int fd, const struct iovec *vector,
                                int count, off_t offset) {
        ssize_t total_write = 0;
        for (int i = 0; i < count; ++i) {
            const ssize_t rc = ::pwrite(fd, vector[i].iov_base, vector[i].iov_len, offset);
            if (rc <= 0) {
                return total_write > 0 ? total_write : rc;
            }
            total_write += rc;
            offset += rc;
            if (rc < (ssize_t) vector[i].iov_len) {
                break;
            }
        }
        return total_write;
    }


#if defined(TURBO_PROCESSOR_X86_64)

#ifndef SYS_preadv
#define SYS_preadv 295
#endif  // SYS_preadv

#ifndef SYS_pwritev
#define SYS_pwritev 296
#endif // SYS_pwritev

    // SYS_preadv/SYS_pwritev is available since Linux 2.6.30
    static ssize_t sys_preadv(int fd, const struct iovec *vector,
                              int count, off_t offset) {
        return syscall(SYS_preadv, fd, vector, count, offset);
    }

    static ssize_t sys_pwritev(int fd, const struct iovec *vector,
                               int count, off_t offset) {
        return syscall(SYS_pwritev, fd, vector, count, offset);
    }

    inline iov_function get_preadv_func() {
#if defined(TURBO_PLATFORM_OSX)
        return user_preadv;
#endif
        turbo::FDGuard fd(::open("/dev/zero", O_RDONLY));
        if (fd < 0) {
            TLOG_WARN("Fail to open /dev/zero");
            return user_preadv;
        }
        char dummy[1];
        iovec vec = { dummy, sizeof(dummy) };
        const int rc = syscall(SYS_preadv, (int)fd, &vec, 1, 0);
        if (rc < 0) {
            TLOG_WARN("The kernel doesn't support SYS_preadv, "
                      " use user_preadv instead");
            return user_preadv;
        }
        return sys_preadv;
    }

    inline iov_function get_pwritev_func() {
        turbo::FDGuard fd(::open("/dev/null", O_WRONLY));
        if (fd < 0) {
            TLOG_ERROR("Fail to open /dev/null");
            return user_pwritev;
        }
#if defined(TURBO_PLATFORM_OSX)
        return user_pwritev;
#endif
        char dummy[1];
        iovec vec = { dummy, sizeof(dummy) };
        const int rc = syscall(SYS_pwritev, (int)fd, &vec, 1, 0);
        if (rc < 0) {
            TLOG_WARN("The kernel doesn't support SYS_pwritev, "
                      " use user_pwritev instead");
            return user_pwritev;
        }
        return sys_pwritev;
    }

#else   // TURBO_PROCESSOR_X86_64

    #warning "We don't check if the kernel supports SYS_preadv or SYS_pwritev on non-X86_64, use implementation on pread/pwrite directly."

        inline iov_function get_preadv_func() {
            return user_preadv;
        }

        inline iov_function get_pwritev_func() {
            return user_pwritev;
        }

#endif  // TURBO_PROCESSOR_X86_64

}  // namespace turbo::system_internal

namespace turbo {

    ssize_t sys_pwritev(FILE_HANDLER fd, const struct iovec *vector, int count, off_t offset) {
        static system_internal::iov_function pwritev_func = system_internal::get_pwritev_func();
        return pwritev_func(fd, vector, count, offset);
    }

    ssize_t sys_preadv(FILE_HANDLER fd, const struct iovec *vector, int count, off_t offset) {
        static system_internal::iov_function preadv_func = system_internal::get_preadv_func();
        return preadv_func(fd, vector, count, offset);
    }

    ssize_t sys_pwrite(FILE_HANDLER fd, const void *data, int count, off_t offset) {
        struct iovec iov = { const_cast<void *>(data), static_cast<size_t>(count) };
        return sys_pwritev(fd, &iov, 1, offset);
    }

    ssize_t sys_pread(FILE_HANDLER fd, const void *data, int count, off_t offset) {
        struct iovec iov = { const_cast<void *>(data), static_cast<size_t>(count) };
        return sys_preadv(fd, &iov, 1, offset);
    }

    ssize_t sys_writev(FILE_HANDLER fd, const struct iovec *vector, int count) {
        return ::writev(fd, vector, count);
    }

    ssize_t sys_readv(FILE_HANDLER fd, const struct iovec *vector, int count) {
        return ::readv(fd, vector, count);
    }


    ssize_t sys_write(FILE_HANDLER fd, const void *data, int count) {
        struct iovec iov = { const_cast<void *>(data), static_cast<size_t>(count) };
        return sys_writev(fd, &iov, 1);
    }

    ssize_t sys_read(FILE_HANDLER fd, const void *data, int count) {
        struct iovec iov = { const_cast<void *>(data), static_cast<size_t>(count) };
        return sys_readv(fd, &iov, 1);
    }

    ResultStatus<FILE_HANDLER> open_file(const std::string &filename, const OpenOption &option) {
        const FILE_HANDLER fd = ::open((filename.c_str()), option.flags, option.mode);
        if (fd == -1) {
            return make_status();
        }
        return fd;
    }

    ResultStatus<FILE_HANDLER> open_file_read(const std::string &filename) {
        return open_file(filename, kDefaultReadOption);
    }

    ResultStatus<FILE_HANDLER> open_file_write(const std::string &filename, bool truncate) {
        if (truncate) {
            return open_file(filename, kDefaultTruncateWriteOption);
        }
        return open_file(filename, kDefaultAppendWriteOption);
    }

}

#ifdef TURBO_PLATFORM_POSIX
namespace turbo {
    ssize_t file_size(int fd) {
        if (fd == -1) {
            return -1;
        }
// 64 bits(but not in osx or cygwin, where fstat64 is deprecated)
#    if (defined(__linux__) || defined(__sun) || defined(_AIX)) && (defined(__LP64__) || defined(_LP64))
        struct stat64 st;
        if (::fstat64(fd, &st) == 0) {
            return static_cast<size_t>(st.st_size);
        }
#    else // other unix or linux 32 bits or cygwin
        struct stat st;
        if (::fstat(fd, &st) == 0) {
            return static_cast<size_t>(st.st_size);
        }
#    endif
        return -1;
    }
}
#endif  // TURBO_PLATFORM_POSIX

#ifdef TURBO_PLATFORM_WINDOWS

namespace turbo {
        ssize_t file_size(int fd) {
        if (fd == -1) {
            return -1;
        }
#    if defined(_WIN64) // 64 bits
                __int64 ret = ::_filelengthi64(fd);
                if (ret >= 0) {
                    return static_cast<size_t>(ret);
                }

#    else // windows 32 bits
                long ret = ::_filelength(fd);
                if (ret >= 0) {
                    return static_cast<size_t>(ret);
                }
#    endif
        return -1;
    }

}
#endif  // TURBO_PLATFORM_WINDOWS
