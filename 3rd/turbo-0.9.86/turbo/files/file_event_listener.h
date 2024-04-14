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

#ifndef TURBO_FILES_FILE_EVENT_LISTENER_H_
#define TURBO_FILES_FILE_EVENT_LISTENER_H_

#include <functional>
#include <cstdio>
#include "turbo/files/internal/filesystem.h"
#include "turbo/system/io.h"

namespace turbo {

     /**
      * @ingroup turbo_files_operation
      * @brief FileEventListener is a listener for file events. It is used to
      *        inspire by RocksDB's EventListener. It is used in the
      *        FileMonitor class. or some condition that you want to trigger
      *        some action during the file operation.
      */
    struct FileEventListener {
        FileEventListener()
                : before_open(nullptr), after_open(nullptr), before_close(nullptr), after_close(nullptr) {}

        std::function<void(const turbo::filesystem::path &filename)> before_open;
        std::function<void(const turbo::filesystem::path &filename, FILE_HANDLER handler)> after_open;
        std::function<void(const turbo::filesystem::path &filename, FILE_HANDLER file_stream)> before_close;
        std::function<void(const turbo::filesystem::path &filename)> after_close;
    };
}  // namespace
#endif  // TURBO_FILES_FILE_EVENT_LISTENER_H_
