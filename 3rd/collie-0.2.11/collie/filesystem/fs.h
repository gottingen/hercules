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

#ifndef COLLIE_FILESYSTEM_FS_H_
#define COLLIE_FILESYSTEM_FS_H_

#include <string_view>
#include <collie/filesystem/ghc/filesystem.hpp>

namespace collie {
    using namespace ghc;

    inline bool has_extension(const std::string_view &filename, const std::string_view &extension) {
        return filename.size() >= extension.size() &&
               filename.compare(filename.size() - extension.size(), extension.size(),
                                extension) == 0;
    }

    inline std::string_view trim_extension(const std::string_view &filename, const std::string_view &extension) {
        if (has_extension(filename, extension)) {
            return filename.substr(0, filename.size() - extension.size());
        } else {
            return filename;
        }
    }

}
#endif  // COLLIE_FILESYSTEM_FS_H_
