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
#include "turbo/module/module_loader.h"
#include "turbo/module/constants.h"
#include "turbo/files/filesystem.h"
#include "turbo/format/format.h"

#ifdef TURBO_PLATFORM_WINDOWS
namespace turbo {

    void* ModuleLoader::get_function_pointer(ModuleHandle handle, const std::string& suffix, const char* symbolName) {
        TURBO_UNUSED(suffix);
        return GetProcAddress(handle, symbolName);
    }

    ModuleHandle ModuleLoader::load_library(const char* library_name) {
        return load_library(library_name);
    }

    void ModuleLoader::unload_library(ModuleHandle handle) { FreeLibrary(handle); }

    std::string ModuleLoader::get_error_message() {
        const char* lpMsgBuf;
        DWORD dw = GetLastError();

        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                      FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR)&lpMsgBuf, 0, NULL);
        std::string error_message(lpMsgBuf);
        return error_message;
    }

}  // namespace turbo
#elif defined(TURBO_PLATFORM_LINUX) || defined(TURBO_PLATFORM_OSX)

#include <dlfcn.h>

namespace turbo {
#if defined(TURBO_PLATFORM_LINUX)
    std::vector<std::string> ModuleLoader::module_name(const std::string& name,const std::string& suffix,
                                          const ModuleVersion& ver) {
        TURBO_UNUSED(suffix);
        const std::string noVerName = std::string(turbo::kModulePrefix) + name + std::string(turbo::kModuleSuffix);
        if (ver != kNullVersion) {
            const std::string soname = format("{}.{}", noVerName, ver.major);

            const std::string fullName = format("{}.{}", noVerName, ver.to_string());
            return {fullName, noVerName + soname, noVerName};
        } else {
            return {noVerName};
        }

    }
#else
    std::vector<std::string> ModuleLoader::module_name(const std::string& name,
                                          const ModuleVersion& ver) {
        const std::string noVerName = kModulePrefix + name + kModuleSuffix;
        if (ver != kNullVersion) {
            const std::string fullName = format("{}.{}.{}", kModulePrefix, ver.to_string(), kModuleSuffix);
            return {fullName, noVerName};
        } else {
            return {noVerName};
        }

    }
#endif
    void *ModuleLoader::get_function_pointer(ModuleHandle handle, const char *symbolName) {
        return ::dlsym(handle, symbolName);
    }

    ModuleHandle ModuleLoader::load_library(const char *library_name) {
        return ::dlopen(library_name, RTLD_LAZY);
    }

    void ModuleLoader::unload_library(ModuleHandle handle) { dlclose(handle); }

    std::string ModuleLoader::get_error_message() {
        char *errMsg = dlerror();
        if (errMsg) { return std::string(errMsg); }
        // constructing std::basic_string from NULL/0 address is
        // invalid and has undefined behavior
        return std::string("No Error");
    }
}  // namespace turbo

#endif

