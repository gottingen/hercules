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

#if defined(TURBO_PLATFORM_WINDOWS)
namespace turbo::sys_io {
    turbo::ResultStatus<FILE_HANDLER> open_read(const turbo::filesystem::path &filename, const std::string &mode,
                                                const FileOption &option) {
#ifdef TURBO_WCHAR_FILENAMES
        fp = ::_wfsopen((filename.c_str()), mode.c_str(), _SH_DENYNO);
#else
        fp = ::_fsopen((filename.c_str()), mode.c_str(), _SH_DENYNO);
#endif
        if (fp == nullptr) {
            return turbo::errno_to_status(errno, "");
        }
        auto file_handle = reinterpret_cast<FILE_HANDLER>(_get_osfhandle(::_fileno(fp)));
        if (option.prevent_child) {

            if (!::SetHandleInformation(file_handle, HANDLE_FLAG_INHERIT, 0)) {
                ::fclose(fp);
                fp = nullptr;
                return turbo::errno_to_status(errno, "");
            }
        }
        return file_handle;
    }

}

#endif  // TURBO_PLATFORM_WINDOWS