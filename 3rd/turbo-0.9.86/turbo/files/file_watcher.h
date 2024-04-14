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
#ifndef TURBO_FILES_FILE_WATCHER_H_
#define TURBO_FILES_FILE_WATCHER_H_

#include <cstdint>                                 // int64_t
#include <string>                                   // std::string
#include "turbo/status/status.h"
namespace turbo {

    /**
     * @ingroup turbo_files_monitor
     * @brief FileWatcher is a class to watch file changes. eg
     *        file creation, file modification, file deletion.
     *        Example:
     *        @code {.cpp}
     *        FileWatcher fw;
     *        fw.init("to_be_watched_file");
     *        ....
     *        if (fw.check_and_consume() > 0) {
     *          // the file is created or updated
     *          ......
     *        }
     *       @endcode
     */
    class FileWatcher {
    public:
        enum Change {
            DELETED = -1,
            UNCHANGED = 0,
            UPDATED = 1,
            CREATED = 2,
        };

        typedef int64_t Timestamp;

        FileWatcher();

        /**
         * @brief Watch file at `file_path', must be called before calling other methods.
         * @param file_path
         * @return return ok_status() if success, otherwise return error status.
         */
        turbo::Status init(const char *file_path);


        /**
         * @brief Watch file at `file_path', must be called before calling other methods.
         * @param file_path
         * @return
         */
        turbo::Status init_from_not_exist(const char *file_path);

        /**
         * @brief Check and consume change of the watched file. Write `last_timestamp'
         *        if it's not nullptr.
         * @param [out] last_timestamp the last timestamp of the file.
         * @return
         *        CREATE    the file is created since last call to this method.
         *        UPDATED   the file is modified since last call.
         *        UNCHANGED the file has no change since last call.
         *        DELETED   the file was deleted since last call.
         *@note If the file is updated too frequently, this method may return
         *      UNCHANGED due to precision of stat(2) and the file system. If the file
         *      is created and deleted too frequently, the event may not be detected.
         */

        Change check_and_consume(Timestamp *last_timestamp = nullptr);

        // Set internal timestamp. User can use this method to make
        // check_and_consume() replay the change.
        void restore(Timestamp timestamp);

        // Get path of watched file
        const char *filepath() const { return _file_path.c_str(); }

    private:
        Change check(Timestamp *new_timestamp) const;

        std::string _file_path;
        Timestamp _last_ts;
    };
}  // namespace turbo

#endif  // TURBO_FILES_FILE_WATCHER_H_
