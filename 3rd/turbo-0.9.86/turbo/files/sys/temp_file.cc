
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include <fcntl.h>                                  // open
#include <unistd.h>                                 // close
#include <stdio.h>                                  // snprintf, vdprintf
#include <cstdlib>                                 // mkstemp
#include <string.h>                                 // strlen 
#include <stdarg.h>                                 // va_list
#include <errno.h>                                  // errno
#include <new>                                      // placement new
#include "turbo/files/sys/temp_file.h"                              // TempFile
#include "turbo//log/logging.h"
#include "turbo/random/random.h"
#include "turbo/concurrent/spinlock.h"

// Initializing array. Needs to be macro.
#define BASE_FILES_TEMP_FILE_PATTERN "temp_file_XXXXXX"

namespace turbo {

    static turbo::BitGen temp_file_bit_gen;
    static turbo::SpinLock temp_file_spin_lock;

    std::string TempFile::generate_temp_file_name(std::string_view prefix, std::string_view ext, size_t bits) {
        std::string gen_name;
        {
            std::lock_guard<SpinLock> lock(temp_file_spin_lock);
            for(size_t i = 0; i < bits; ++i) {
                gen_name.push_back(turbo::uniform(temp_file_bit_gen, 'a', 'z'));
            }
        }
        std::string result;
        if(!ext.empty()) {
            result = turbo::format("{}{}.{}", prefix, gen_name, ext);
        } else {
            result = turbo::format("{}{}", prefix, gen_name);
        }
        return result;
    }

    TempFile::TempFile(const FileEventListener &listener) :_file(listener) {

    }

    [[nodiscard]] turbo::Status TempFile::open(std::string_view prefix, std::string_view ext, size_t bits) noexcept {
        if(_ever_opened) {
            return turbo::ok_status();
        }
        _file_path = generate_temp_file_name(prefix, ext, bits);
        auto rs = _file.open(_file_path,kDefaultTruncateWriteOption);
        if(!rs.ok()) {
            return rs;
        }
        _ever_opened = true;
        return turbo::ok_status();
    }

    // Write until all buffer was written or an error except EINTR.
    // Returns:
    //    -1   error happened, errno is set
    // count   all written
    static ssize_t temp_file_write_all(int fd, const void *buf, size_t count) {
        size_t off = 0;
        for (;;) {
            ssize_t nw = write(fd, (char *) buf + off, count - off);
            if (nw == (ssize_t) (count - off)) {  // including count==0
                return count;
            }
            if (nw >= 0) {
                off += nw;
            } else if (errno != EINTR) {
                return -1;
            }
        }
    }

    turbo::Status TempFile::write(const void *buf, size_t count) {
        if(!_ever_opened) {
            return turbo::make_status(kEBADFD);
        }
        return _file.write(buf, count);
    }


} // namespace flare
