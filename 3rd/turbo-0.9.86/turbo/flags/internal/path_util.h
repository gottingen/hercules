// Copyright 2023 The titan-search Authors.
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
#ifndef TURBO_FLAGS_INTERNAL_PATH_UTIL_H_
#define TURBO_FLAGS_INTERNAL_PATH_UTIL_H_

#include "turbo/platform/port.h"
#include "turbo/strings/string_view.h"

namespace turbo::flags_internal {
    // A portable interface that returns the basename of the filename passed as an
    // argument. It is similar to basename(3)
    // <https://linux.die.net/man/3/basename>.
    // For example:
    //     flags_internal::Basename("a/b/prog/file.cc")
    // returns "file.cc"
    //     flags_internal::Basename("file.cc")
    // returns "file.cc"
    inline std::string_view Basename(std::string_view filename) {
        auto last_slash_pos = filename.find_last_of("/\\");

        return last_slash_pos == std::string_view::npos
               ? filename
               : filename.substr(last_slash_pos + 1);
    }

    // A portable interface that returns the directory name of the filename
    // passed as an argument, including the trailing slash.
    // Returns the empty string if a slash is not found in the input file name.
    // For example:
    //      flags_internal::Package("a/b/prog/file.cc")
    // returns "a/b/prog/"
    //      flags_internal::Package("file.cc")
    // returns ""
    inline std::string_view Package(std::string_view filename) {
        auto last_slash_pos = filename.find_last_of("/\\");

        return last_slash_pos == std::string_view::npos
               ? std::string_view()
               : filename.substr(0, last_slash_pos + 1);
    }

}  // namespace turbo::flags_internal

#endif  // TURBO_FLAGS_INTERNAL_PATH_UTIL_H_
