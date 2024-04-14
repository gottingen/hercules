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
#include "turbo/files/file_watcher.h"
#include "turbo/platform/port.h" // TURBO_PLATFORM_OSX
#include <sys/stat.h>

namespace turbo {

    static const FileWatcher::Timestamp NON_EXIST_TS =
            static_cast<FileWatcher::Timestamp>(-1);

    FileWatcher::FileWatcher() : _last_ts(NON_EXIST_TS) {
    }

    turbo::Status FileWatcher::init(const char *file_path) {
        auto rs = init_from_not_exist(file_path);
        if (!rs.ok()) {
            return rs;
        }
        check_and_consume(nullptr);
        return turbo::ok_status();
    }

    turbo::Status FileWatcher::init_from_not_exist(const char *file_path) {
        if (nullptr == file_path) {
            return turbo::make_status(kEINVAL);
        }
        if (!_file_path.empty()) {
            return turbo::make_status(kEINVAL);
        }
        _file_path = file_path;
        return ok_status();
    }

    FileWatcher::Change FileWatcher::check(Timestamp *new_timestamp) const {
        struct stat st;
        const int ret = stat(_file_path.c_str(), &st);
        if (ret < 0) {
            *new_timestamp = NON_EXIST_TS;
            if (NON_EXIST_TS != _last_ts) {
                return DELETED;
            } else {
                return UNCHANGED;
            }
        } else {
            // Use microsecond timestamps which can be used for:
            //   2^63 / 1000000 / 3600 / 24 / 365 = 292471 years
            const Timestamp cur_ts =
#if defined(TURBO_PLATFORM_OSX)
                    st.st_mtimespec.tv_sec * 1000000L + st.st_mtimespec.tv_nsec / 1000L;
#else
                    st.st_mtim.tv_sec * 1000000L + st.st_mtim.tv_nsec / 1000L;
#endif
            *new_timestamp = cur_ts;
            if (NON_EXIST_TS != _last_ts) {
                if (cur_ts != _last_ts) {
                    return UPDATED;
                } else {
                    return UNCHANGED;
                }
            } else {
                return CREATED;
            }
        }
    }

    FileWatcher::Change FileWatcher::check_and_consume(Timestamp *last_timestamp) {
        Timestamp new_timestamp;
        Change e = check(&new_timestamp);
        if (last_timestamp) {
            *last_timestamp = _last_ts;
        }
        if (e != UNCHANGED) {
            _last_ts = new_timestamp;
        }
        return e;
    }

    void FileWatcher::restore(Timestamp timestamp) {
        _last_ts = timestamp;
    }

}  // namespace turbo
