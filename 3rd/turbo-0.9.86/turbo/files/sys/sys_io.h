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

#ifndef TURBO_FILES_INTERNAL_SYS_IO_H_
#define TURBO_FILES_INTERNAL_SYS_IO_H_

#include <string>
#include <cstdio>
#include <tuple>
#include "turbo/files/file_option.h"
#include "turbo/files/internal/filesystem.h"
#include "turbo/files/fwd.h"
#include "turbo/status/result_status.h"

namespace turbo::sys_io {

    turbo::ResultStatus<FILE_HANDLER> open_read(const turbo::filesystem::path &filename, const std::string &mode,
                                                    const FileOption &option = FileOption());

    turbo::ResultStatus<FILE_HANDLER> open_write(const turbo::filesystem::path &filename, const std::string &mode,
                                                     const FileOption &option = FileOption());

}  // namespace turbo::sys_io

#define INVALID_FD_RETURN(fd) \
    if ((fd) == INVALID_FILE_HANDLER) { \
        return turbo::make_status(kEBADFD, "file not open for read yet"); \
    }

#endif // TURBO_FILES_INTERNAL_SYS_IO_H_
