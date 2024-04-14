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

#include "turbo/status/turbo_module.h"
#include "turbo/platform/port.h"
#include <cstdlib> // EXIT_FAILURE
#include <cstdio>  // snprintf
#include <cstring> // strerror_r
#include <mutex>

namespace turbo {

    const int INDEX_BEGIN = 0;
    const int INDEX_END = 4096;
    static const char *module_desc[INDEX_END - INDEX_BEGIN] = {};
    static std::mutex modify_desc_mutex;

    const size_t MODULE_BUFSIZE = 64;
    __thread char tls_module_buf[MODULE_BUFSIZE];

    int DescribeCustomizedModule(
            int module_index, const char *module_name, const char *description) {
        std::unique_lock<std::mutex> l(modify_desc_mutex);
        if (module_index < INDEX_BEGIN || module_index >= INDEX_END) {
            // error() is a non-portable GNU extension that should not be used.
            fprintf(stderr, "Fail to define module %s(%d) which is out of range, abort.",
                    module_name, module_index);
            std::exit(1);
        }
        const char *desc = module_desc[module_index - INDEX_BEGIN];
        if (desc) {
            if (strcmp(desc, description) == 0) {
                fprintf(stderr, "WARNING: Detected shared library loading\n");
                return -1;
            }
        }
        module_desc[module_index - INDEX_BEGIN] = description;
        return 0;  // must
    }

}  // namespace turbo

const char *TurboModule(int module_index) {
    if (module_index >= turbo::INDEX_BEGIN && module_index < turbo::INDEX_END) {
        const char *s = turbo::module_desc[module_index - turbo::INDEX_BEGIN];
        if (s) {
            return s;
        }
    }
    snprintf(turbo::tls_module_buf, turbo::MODULE_BUFSIZE - 1, "%d_UDM", module_index);
    return turbo::tls_module_buf;
}

TURBO_REGISTER_MODULE_INDEX(turbo::kTurboModuleIndex, "TURBO");