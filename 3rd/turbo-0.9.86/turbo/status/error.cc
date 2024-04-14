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

#include "turbo/status/error.h"
#include <cstdlib> // EXIT_FAILURE
#include <cstdio>  // snprintf
#include <mutex>

namespace turbo {

    static std::mutex modify_desc_mutex;

    std::array<const char*, ERRNO_END - ERRNO_BEGIN> errno_desc_array = {};

    int describe_customized_errno(
            int error_code, const char *error_name, const char *description) {
        if (description == nullptr) {
            ::fprintf(stderr, "description is nullptr, abort.");
            std::abort();
        }
        std::unique_lock<std::mutex> l(modify_desc_mutex);
        if (error_code < ERRNO_BEGIN || error_code >= ERRNO_END) {
            // error() is a non-portable GNU extension that should not be used.
            ::fprintf(stderr, "Fail to define %s(%d) which is out of range, abort.",
                      error_name, error_code);
            std::abort();
        }
        const auto index = error_code - ERRNO_BEGIN;
        const char* desc = errno_desc_array[index];
        if (desc != nullptr) {
            ::fprintf(stderr, "WARNING: Detected shared library loading\n");
            std::abort();
        }
        errno_desc_array[index] = description;
        return 0;  // must
    }

}  // namespace turbo
