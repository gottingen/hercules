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

#include <hercules/hir/base.h>

#include <hercules/hir/types/types.h>
#include <hercules/hir/util/format.h>
#include <hercules/hir/value.h>
#include <hercules/hir/var.h>

namespace hercules::ir {

    id_t IdMixin::currentId = 0;

    void IdMixin::resetId() { currentId = 0; }

    const char Node::NodeId = 0;

    std::ostream &operator<<(std::ostream &os, const Node &other) {
        return util::format(os, &other);
    }

    int Node::replaceUsedValue(Value *old, Value *newValue) {
        return replaceUsedValue(old->getId(), newValue);
    }

    int Node::replaceUsedType(types::Type *old, types::Type *newType) {
        return replaceUsedType(old->getName(), newType);
    }

    int Node::replaceUsedVariable(Var *old, Var *newVar) {
        return replaceUsedVariable(old->getId(), newVar);
    }

} // namespace hercules::ir
