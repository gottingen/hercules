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

#include <memory>
#include <string>
#include <vector>

#include "hercules/parser/ast/types/type.h"
#include "hercules/parser/visitors/format/format.h"
#include "hercules/parser/visitors/typecheck/typecheck.h"

namespace hercules::ast::types {

/// Undo a destructive unification.
void Type::Unification::undo() {
  for (size_t i = linked.size(); i-- > 0;) {
    linked[i]->kind = LinkType::Unbound;
    linked[i]->type = nullptr;
  }
  for (size_t i = leveled.size(); i-- > 0;) {
    seqassertn(leveled[i].first->kind == LinkType::Unbound, "not unbound [{}]",
               leveled[i].first->getSrcInfo());
    leveled[i].first->level = leveled[i].second;
  }
  for (auto &t : traits)
    t->trait = nullptr;
}

Type::Type(const std::shared_ptr<Type> &typ) : cache(typ->cache) {
  setSrcInfo(typ->getSrcInfo());
}

Type::Type(Cache *cache, const SrcInfo &info) : cache(cache) { setSrcInfo(info); }

TypePtr Type::follow() { return shared_from_this(); }

std::vector<std::shared_ptr<Type>> Type::getUnbounds() const { return {}; }

std::string Type::toString() const { return debugString(1); }

std::string Type::prettyString() const { return debugString(0); }

bool Type::is(const std::string &s) { return getClass() && getClass()->name == s; }

char Type::isStaticType() {
  auto t = follow();
  if (auto s = t->getStatic())
    return char(s->expr->staticValue.type);
  if (auto l = t->getLink())
    return l->isStatic;
  return false;
}

TypePtr Type::makeType(Cache *cache, const std::string &name,
                       const std::string &niceName, bool isRecord) {
  if (name == "Union")
    return std::make_shared<UnionType>(cache);
  if (isRecord)
    return std::make_shared<RecordType>(cache, name, niceName);
  return std::make_shared<ClassType>(cache, name, niceName);
}

std::shared_ptr<StaticType> Type::makeStatic(Cache *cache, const ExprPtr &expr) {
  return std::make_shared<StaticType>(cache, expr);
}

} // namespace hercules::ast::types
