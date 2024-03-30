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

#include <hercules/parser/ast/types/class.h>

namespace hercules::ast {
    struct StaticValue;
}

namespace hercules::ast::types {

    /**
     * A static integer type (e.g. N in def foo[N: int]). Usually an integer, but can point
     * to a static expression.
     */
    struct StaticType : public Type {
        /// List of static variables that a type depends on
        /// (e.g. for A+B+2, generics are {A, B}).
        std::vector<ClassType::Generic> generics;
        /// A static expression that needs to be evaluated.
        /// Can be nullptr if there is no expression.
        std::shared_ptr<Expr> expr;

        StaticType(Cache *cache, std::vector<ClassType::Generic> generics,
                   const std::shared_ptr<Expr> &expr);

        /// Convenience function that parses expr and populates static type generics.
        StaticType(Cache *cache, const std::shared_ptr<Expr> &expr);

        /// Convenience function for static types whose evaluation is already known.
        explicit StaticType(Cache *cache, int64_t i);

        explicit StaticType(Cache *cache, const std::string &s);

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

        StaticValue evaluate() const;

        std::shared_ptr<StaticType> getStatic() override {
            return std::static_pointer_cast<StaticType>(shared_from_this());
        }

    private:
        void parseExpr(const std::shared_ptr<Expr> &e, std::unordered_set<std::string> &seen);
    };

} // namespace hercules::ast::types
