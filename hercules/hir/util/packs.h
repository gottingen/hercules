// Copyright 2024 The EA Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#pragma once

#include <vector>

namespace hercules::ir::util {


    /// Utility function to strip parameter packs.
    /// @param dst the destination vector
    /// @param first the value
    template<typename Desired>
    void stripPack(std::vector<Desired *> &dst, Desired &first) {
        dst.push_back(&first);
    }

    /// Utility function to strip parameter packs.
    /// @param dst the destination vector
    template<typename Desired>
    void stripPack(std::vector<Desired *> &dst) {}

    /// Utility function to strip parameter packs.
    /// @param dst the destination vector
    /// @param first the value
    /// @param args the argument pack
    template<typename Desired, typename... Args>
    void stripPack(std::vector<Desired *> &dst, Desired &first, Args &&...args) {
        dst.push_back(&first);
        stripPack<Desired>(dst, std::forward<Args>(args)...);
    }

} // namespace hercules::ir::util
