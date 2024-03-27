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

#include <iostream>

#include "hercules/hir/cir.h"

namespace hercules::ir::util {

    /// Formats an IR node.
    /// @param node the node
    /// @return the formatted node
    std::string format(const Node *node);

    /// Formats an IR node to an IO stream.
    /// @param os the output stream
    /// @param node the node
    /// @return the resulting output stream
    std::ostream &format(std::ostream &os, const Node *node);

} // namespace hercules::ir::util
