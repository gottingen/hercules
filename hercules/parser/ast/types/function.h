// Copyright 2023 The titan-search Authors.
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
#include "hercules/parser/ast/types/type.h"

namespace hercules::ast {
    struct FunctionStmt;
}

namespace hercules::ast::types {

    /**
     * A generic type that represents a Seq function instantiation.
     * It inherits RecordType that realizes Callable[...].
     *
     * ⚠️ This is not a function pointer (Function[...]) type.
     */
    struct FuncType : public RecordType {
        /// Canonical AST node.
        FunctionStmt *ast;
        /// Function generics (e.g. T in def foo[T](...)).
        std::vector<ClassType::Generic> funcGenerics;
        /// Enclosing class or a function.
        TypePtr funcParent;

    public:
        FuncType(
                const std::shared_ptr<RecordType> &baseType, FunctionStmt *ast,
                std::vector<ClassType::Generic> funcGenerics = std::vector<ClassType::Generic>(),
                TypePtr funcParent = nullptr);

    public:
        int unify(Type *typ, Unification *undo) override;

        TypePtr generalize(int atLevel) override;

        TypePtr instantiate(int atLevel, int *unboundCount,
                            std::unordered_map<int, TypePtr> *cache) override;

    public:
        std::vector<TypePtr> getUnbounds() const override;

        bool canRealize() const override;

        bool isInstantiated() const override;

        std::string debugString(char mode) const override;

        std::string realizedName() const override;

        std::string realizedTypeName() const override;

        std::shared_ptr<FuncType> getFunc() override {
            return std::static_pointer_cast<FuncType>(shared_from_this());
        }

        std::vector<TypePtr> &getArgTypes() const {
            return generics[0].type->getRecord()->args;
        }

        TypePtr getRetType() const { return generics[1].type; }
    };

    using FuncTypePtr = std::shared_ptr<FuncType>;

    /**
     * A generic type that represents a partial Seq function instantiation.
     * It inherits RecordType that realizes Tuple[...].
     *
     * Note: partials only work on Seq functions. Function pointer partials
     *       will become a partials of Function.__call__ Seq function.
     */
    struct PartialType : public RecordType {
        /// Seq function that is being partialized. Always generic (not instantiated).
        FuncTypePtr func;
        /// Arguments that are already provided (1 for known argument, 0 for expecting).
        std::vector<char> known;

    public:
        PartialType(const std::shared_ptr<RecordType> &baseType,
                    std::shared_ptr<FuncType> func, std::vector<char> known);

    public:
        int unify(Type *typ, Unification *us) override;

        TypePtr generalize(int atLevel) override;

        TypePtr instantiate(int atLevel, int *unboundCount,
                            std::unordered_map<int, TypePtr> *cache) override;

        std::string debugString(char mode) const override;

        std::string realizedName() const override;

    public:
        std::shared_ptr<PartialType> getPartial() override {
            return std::static_pointer_cast<PartialType>(shared_from_this());
        }
    };

    using PartialTypePtr = std::shared_ptr<PartialType>;

} // namespace hercules::ast::types
