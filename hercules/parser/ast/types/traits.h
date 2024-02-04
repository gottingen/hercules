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
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "hercules/parser/ast/types/type.h"

namespace hercules::ast::types {

struct Trait : public Type {
  bool canRealize() const override;
  bool isInstantiated() const override;
  std::string realizedName() const override;

protected:
  explicit Trait(const std::shared_ptr<Type> &);
  explicit Trait(Cache *);
};

struct CallableTrait : public Trait {
  std::vector<TypePtr> args; // tuple with arg types, ret type

public:
  explicit CallableTrait(Cache *cache, std::vector<TypePtr> args);
  int unify(Type *typ, Unification *undo) override;
  TypePtr generalize(int atLevel) override;
  TypePtr instantiate(int atLevel, int *unboundCount,
                      std::unordered_map<int, TypePtr> *cache) override;
  std::string debugString(char mode) const override;
};

struct TypeTrait : public Trait {
  TypePtr type;

public:
  explicit TypeTrait(TypePtr type);
  int unify(Type *typ, Unification *undo) override;
  TypePtr generalize(int atLevel) override;
  TypePtr instantiate(int atLevel, int *unboundCount,
                      std::unordered_map<int, TypePtr> *cache) override;
  std::string debugString(char mode) const override;
};

struct VariableTupleTrait : public Trait {
  TypePtr size;

public:
  explicit VariableTupleTrait(TypePtr size);
  int unify(Type *typ, Unification *undo) override;
  TypePtr generalize(int atLevel) override;
  TypePtr instantiate(int atLevel, int *unboundCount,
                      std::unordered_map<int, TypePtr> *cache) override;
  std::string debugString(char mode) const override;
};

} // namespace hercules::ast::types
