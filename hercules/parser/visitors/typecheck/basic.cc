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

#include "hercules/parser/ast.h"
#include "hercules/parser/cache.h"
#include "hercules/parser/common.h"
#include "hercules/parser/visitors/simplify/simplify.h"
#include "hercules/parser/visitors/typecheck/typecheck.h"

using fmt::format;

namespace hercules::ast {

using namespace types;

/// Set type to `Optional[?]`
void TypecheckVisitor::visit(NoneExpr *expr) {
  unify(expr->type, ctx->instantiate(ctx->getType(TYPE_OPTIONAL)));
  if (realize(expr->type)) {
    // Realize the appropriate `Optional.__new__` for the translation stage
    auto cls = expr->type->getClass();
    auto f = ctx->forceFind(TYPE_OPTIONAL ".__new__:0")->type;
    auto t = realize(ctx->instantiate(f, cls)->getFunc());
    expr->setDone();
  }
}

/// Set type to `bool`
void TypecheckVisitor::visit(BoolExpr *expr) {
  unify(expr->type, ctx->getType("bool"));
  expr->setDone();
}

/// Set type to `int`
void TypecheckVisitor::visit(IntExpr *expr) {
  unify(expr->type, ctx->getType("int"));
  expr->setDone();
}

/// Set type to `float`
void TypecheckVisitor::visit(FloatExpr *expr) {
  unify(expr->type, ctx->getType("float"));
  expr->setDone();
}

/// Set type to `str`
void TypecheckVisitor::visit(StringExpr *expr) {
  unify(expr->type, ctx->getType("str"));
  expr->setDone();
}

} // namespace hercules::ast
