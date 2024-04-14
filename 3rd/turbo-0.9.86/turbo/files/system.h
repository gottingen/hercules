//
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
#ifndef TURBO_FILES_SYSTEM_H_
#define TURBO_FILES_SYSTEM_H_

#include "turbo/platform/port.h"

#if defined(TURBO_PLATFORM_WINDOWS)
#  if defined(NOMINMAX)
#    include <windows.h>
#  else
#    define NOMINMAX
#    include <windows.h>
#    undef NOMINMAX
#  endif
#endif
#ifdef TURBO_PLATFORM_OSX
#  include <cstdint>
#  include <mach-o/dyld.h>
#endif
#if defined(TURBO_PLATFORM_POSIX)

#include <unistd.h>
#include <sys/types.h>
#endif

#include <cstdlib>
#include <cstring>
#include <string>

namespace turbo {
    std::string executable_path();

    std::string prefix_path();

    /******************
     * implementation *
     ******************/

    inline std::string executable_path() {
        std::string path;
        char buffer[1024];
        std::memset(buffer, '\0', sizeof(buffer));
#if defined(TURBO_PLATFORM_LINUX)
        if (readlink("/proc/self/exe", buffer, sizeof(buffer)) != -1) {
            path = buffer;
        } else {
            // failed to determine run path
        }
#elif defined (TURBO_PLATFORM_WINDOWS)
        if (GetModuleFileName(nullptr, buffer, sizeof(buffer)) != 0)
        {
            path = buffer;
        }
        else
        {
            // failed to determine run path
        }
#elif defined (TURBO_PLATFORM_OSX)
        std::uint32_t size = sizeof(buffer);
        if(_NSGetExecutablePath(buffer, &size) == 0)
        {
            path = buffer;
        }
        else
        {
            // failed to determine run path
        }
#endif
        return path;
    }

    inline std::string prefix_path() {
        std::string path = executable_path();
#if defined (TURBO_PLATFORM_WINDOWS)
        char separator = '\\';
#else
        char separator = '/';
#endif
        std::string bin_folder = path.substr(0, path.find_last_of(separator));
        std::string prefix = bin_folder.substr(0, bin_folder.find_last_of(separator)) + separator;
        return prefix;
    }
}

#endif  // TURBO_FILES_SYSTEM_H_

