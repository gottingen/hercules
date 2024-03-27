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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hercules/parser/ast/types/class.h"

namespace hercules::ast::types {

    struct UnionType : public RecordType {
        std::vector<TypePtr> pendingTypes;

        explicit UnionType(Cache *cache);

        UnionType(Cache *, const std::vector<ClassType::Generic> &,
                  const std::vector<TypePtr> &);

    public:
        int unify(Type *typ, Unification *undo) override;

        TypePtr generalize(int atLevel) override;

        TypePtr instantiate(int atLevel, int *unboundCount,
                            std::unordered_map<int, TypePtr> *cache) override;

    public:
        bool canRealize() const override;

        std::string debugString(char mode) const override;

        std::string realizedName() const override;

        std::string realizedTypeName() const override;

        bool isSealed() const;

        std::shared_ptr<UnionType> getUnion() override {
            return std::static_pointer_cast<UnionType>(shared_from_this());
        }

        void addType(TypePtr typ);

        void seal();

        std::vector<types::TypePtr> getRealizationTypes();
    };

} // namespace hercules::ast::types
