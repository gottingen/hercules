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

#include "hercules/parser/ast/types/traits.h"
#include "hercules/parser/ast/types/type.h"

namespace hercules::ast::types {

struct LinkType : public Type {
  /// Enumeration describing the current state.
  enum Kind { Unbound, Generic, Link } kind;
  /// The unique identifier of an unbound or generic type.
  int id;
  /// The type-checking level of an unbound type.
  int level;
  /// The type to which LinkType points to. nullptr if unknown (unbound or generic).
  TypePtr type;
  /// >0 if a type is a static type (e.g. N in Int[N: int]); 0 otherwise.
  char isStatic;
  /// Optional trait that unbound type requires prior to unification.
  std::shared_ptr<Trait> trait;
  /// The generic name of a generic type, if applicable. Used for pretty-printing.
  std::string genericName;
  /// Type that will be used if an unbound is not resolved.
  TypePtr defaultType;

public:
  LinkType(Cache *cache, Kind kind, int id, int level = 0, TypePtr type = nullptr,
           char isStatic = 0, std::shared_ptr<Trait> trait = nullptr,
           TypePtr defaultType = nullptr, std::string genericName = "");
  /// Convenience constructor for linked types.
  explicit LinkType(TypePtr type);

public:
  int unify(Type *typ, Unification *undodo) override;
  TypePtr generalize(int atLevel) override;
  TypePtr instantiate(int atLevel, int *unboundCount,
                      std::unordered_map<int, TypePtr> *cache) override;

public:
  TypePtr follow() override;
  std::vector<TypePtr> getUnbounds() const override;
  bool canRealize() const override;
  bool isInstantiated() const override;
  std::string debugString(char mode) const override;
  std::string realizedName() const override;

  std::shared_ptr<LinkType> getLink() override;
  std::shared_ptr<FuncType> getFunc() override;
  std::shared_ptr<PartialType> getPartial() override;
  std::shared_ptr<ClassType> getClass() override;
  std::shared_ptr<RecordType> getRecord() override;
  std::shared_ptr<StaticType> getStatic() override;
  std::shared_ptr<UnionType> getUnion() override;
  std::shared_ptr<LinkType> getUnbound() override;

private:
  /// Checks if a current (unbound) type occurs within a given type.
  /// Needed to prevent a recursive unification (e.g. ?1 with list[?1]).
  bool occurs(Type *typ, Type::Unification *undo);
};

} // namespace hercules::ast::types
