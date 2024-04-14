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

#ifndef TURBO_SYMBOL_SYMBOL_LOADER_H_
#define TURBO_SYMBOL_SYMBOL_LOADER_H_

#include "turbo/platform/port.h"
#include <string>
#include <vector>
#include "turbo/module/module_version.h"
#include "turbo/module/constants.h"

#ifdef TURBO_PLATFORM_WINDOWS
#include <windows.h>
#endif

namespace turbo {

#ifdef TURBO_PLATFORM_WINDOWS
    using ModuleHandle = HMODULE;
#elif defined(TURBO_PLATFORM_LINUX) || defined(TURBO_PLATFORM_OSX)
    using ModuleHandle = void *;
#else
#error "Unsupported platform"
#endif

    struct ModuleLoader {

        static std::vector<std::string> module_name(const std::string& name, const std::string& suffix = "", const ModuleVersion& ver = kNullVersion);

        static void *get_function_pointer(ModuleHandle handle, const char *symbolName);

        static ModuleHandle load_library(const char *library_name);

        static void unload_library(ModuleHandle handle);

        static std::string get_error_message();
    };

}  // namespace turbo

#endif  // TURBO_SYMBOL_SYMBOL_LOADER_H_
