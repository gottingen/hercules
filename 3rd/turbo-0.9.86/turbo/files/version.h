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


#ifndef TURBO_FILES_VERSION_H_
#define TURBO_FILES_VERSION_H_

#include "turbo/module/module_version.h"

/**
 * @defgroup turbo_files filesystem - A C++17-like filesystem implementation for
 * @defgroup turbo_files_filesystem filesystem - A C++17-like filesystem implementation for
 * @defgroup turbo_files_read_file read_file - A function to read a file into a string
 * @defgroup turbo_files_write_file write_file - A function to write a string to a file
 * @defgroup turbo_files_operation operation - A function to perform file operations
 * @defgroup turbo_files_monitor monitor - A class to monitor file changes
 * @defgroup turbo_files_utility utility - A class to provide some file utilities
 */
namespace turbo {
    static constexpr turbo::ModuleVersion files_version = turbo::ModuleVersion{0, 9, 36};
}  // namespace turbo
#endif  // TURBO_FILES_VERSION_H_
