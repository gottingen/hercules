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

#include "hercules/hir/var.h"
#include "hercules/hir/module.h"

namespace hercules::ir {

    const char Var::NodeId = 0;

    int Var::doReplaceUsedType(const std::string &name, types::Type *newType) {
        if (type->getName() == name) {
            type = newType;
            return 1;
        }
        return 0;
    }

    const char VarValue::NodeId = 0;

    int VarValue::doReplaceUsedVariable(id_t id, Var *newVar) {
        if (val->getId() == id) {
            val = newVar;
            return 1;
        }
        return 0;
    }

    const char PointerValue::NodeId = 0;

    types::Type *PointerValue::doGetType() const {
        return getModule()->getPointerType(val->getType());
    }

    int PointerValue::doReplaceUsedVariable(id_t id, Var *newVar) {
        if (val->getId() == id) {
            val = newVar;
            return 1;
        }
        return 0;
    }

} // namespace hercules::ir
