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

#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hercules/cir/cir.h"
#include "hercules/parser/ast.h"
#include "hercules/parser/cache.h"
#include "hercules/parser/common.h"
#include "hercules/parser/visitors/translate/translate_ctx.h"
#include "hercules/parser/visitors/visitor.h"

namespace hercules::ast {

    class TranslateVisitor : public CallbackASTVisitor<ir::Value *, ir::Value *> {
        std::shared_ptr<TranslateContext> ctx;
        ir::Value *result;

    public:
        explicit TranslateVisitor(std::shared_ptr<TranslateContext> ctx);

        static hercules::ir::Func *apply(Cache *cache, const StmtPtr &stmts);

        ir::Value *transform(const ExprPtr &expr) override;

        ir::Value *transform(const StmtPtr &stmt) override;

    private:
        void defaultVisit(Expr *expr) override;

        void defaultVisit(Stmt *expr) override;

    public:
        void visit(NoneExpr *) override;

        void visit(BoolExpr *) override;

        void visit(IntExpr *) override;

        void visit(FloatExpr *) override;

        void visit(StringExpr *) override;

        void visit(IdExpr *) override;

        void visit(IfExpr *) override;

        void visit(CallExpr *) override;

        void visit(DotExpr *) override;

        void visit(YieldExpr *) override;

        void visit(StmtExpr *) override;

        void visit(PipeExpr *) override;

        void visit(SuiteStmt *) override;

        void visit(BreakStmt *) override;

        void visit(ContinueStmt *) override;

        void visit(ExprStmt *) override;

        void visit(AssignStmt *) override;

        void visit(AssignMemberStmt *) override;

        void visit(ReturnStmt *) override;

        void visit(YieldStmt *) override;

        void visit(WhileStmt *) override;

        void visit(ForStmt *) override;

        void visit(IfStmt *) override;

        void visit(TryStmt *) override;

        void visit(ThrowStmt *) override;

        void visit(FunctionStmt *) override;

        void visit(ClassStmt *) override;

        void visit(CommentStmt *) override {}

    private:
        ir::types::Type *getType(const types::TypePtr &t);

        void transformFunctionRealizations(const std::string &name, bool isLLVM);

        void transformFunction(types::FuncType *type, FunctionStmt *ast, ir::Func *func);

        void transformLLVMFunction(types::FuncType *type, FunctionStmt *ast, ir::Func *func);

        template<typename ValueType, typename... Args>
        ValueType *make(Args &&...args) {
            auto *ret = ctx->getModule()->N<ValueType>(std::forward<Args>(args)...);
            return ret;
        }
    };

} // namespace hercules::ast
