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

#include <ostream>
#include <string>
#include <vector>

#include <hercules/parser/ast.h>
#include <hercules/parser/cache.h>
#include <hercules/parser/common.h>
#include <hercules/parser/visitors/visitor.h>
#include <collie/strings/format.h>

namespace hercules::ast {

    class FormatVisitor : public CallbackASTVisitor<std::string, std::string> {
        std::string result;
        std::string space;
        bool renderType, renderHTML;
        int indent;

        std::string header, footer, nl;
        std::string typeStart, typeEnd;
        std::string nodeStart, nodeEnd;
        std::string exprStart, exprEnd;
        std::string commentStart, commentEnd;
        std::string keywordStart, keywordEnd;

        Cache *cache;

    private:
        template<typename T, typename... Ts>
        std::string renderExpr(T &&t, Ts &&...args) {
            std::string s;
            return collie::format("{}{}{}{}{}{}", exprStart, s, nodeStart, collie::format(args...),
                               nodeEnd, exprEnd);
        }

        template<typename... Ts>
        std::string renderComment(Ts &&...args) {
            return collie::format("{}{}{}", commentStart, collie::format(args...), commentEnd);
        }

        std::string pad(int indent = 0) const;

        std::string newline() const;

        std::string keyword(const std::string &s) const;

    public:
        FormatVisitor(bool html, Cache *cache = nullptr);

        std::string transform(const ExprPtr &e) override;

        std::string transform(const Expr *expr);

        std::string transform(const StmtPtr &stmt) override;

        std::string transform(Stmt *stmt, int indent);

        template<typename T>
        static std::string apply(const T &stmt, Cache *cache = nullptr, bool html = false,
                                 bool init = false) {
            auto t = FormatVisitor(html, cache);
            return collie::format("{}{}{}", t.header, t.transform(stmt), t.footer);
        }

        void defaultVisit(Expr *e) override { error("cannot format {}", *e); }

        void defaultVisit(Stmt *e) override { error("cannot format {}", *e); }

    public:
        void visit(NoneExpr *) override;

        void visit(BoolExpr *) override;

        void visit(IntExpr *) override;

        void visit(FloatExpr *) override;

        void visit(StringExpr *) override;

        void visit(IdExpr *) override;

        void visit(StarExpr *) override;

        void visit(KeywordStarExpr *) override;

        void visit(TupleExpr *) override;

        void visit(ListExpr *) override;

        void visit(SetExpr *) override;

        void visit(DictExpr *) override;

        void visit(GeneratorExpr *) override;

        void visit(DictGeneratorExpr *) override;

        void visit(InstantiateExpr *expr) override;

        void visit(IfExpr *) override;

        void visit(UnaryExpr *) override;

        void visit(BinaryExpr *) override;

        void visit(PipeExpr *) override;

        void visit(IndexExpr *) override;

        void visit(CallExpr *) override;

        void visit(DotExpr *) override;

        void visit(SliceExpr *) override;

        void visit(EllipsisExpr *) override;

        void visit(LambdaExpr *) override;

        void visit(YieldExpr *) override;

        void visit(StmtExpr *expr) override;

        void visit(AssignExpr *expr) override;

        void visit(SuiteStmt *) override;

        void visit(BreakStmt *) override;

        void visit(ContinueStmt *) override;

        void visit(ExprStmt *) override;

        void visit(AssignStmt *) override;

        void visit(AssignMemberStmt *) override;

        void visit(DelStmt *) override;

        void visit(PrintStmt *) override;

        void visit(ReturnStmt *) override;

        void visit(YieldStmt *) override;

        void visit(AssertStmt *) override;

        void visit(WhileStmt *) override;

        void visit(ForStmt *) override;

        void visit(IfStmt *) override;

        void visit(MatchStmt *) override;

        void visit(ImportStmt *) override;

        void visit(TryStmt *) override;

        void visit(GlobalStmt *) override;

        void visit(ThrowStmt *) override;

        void visit(FunctionStmt *) override;

        void visit(ClassStmt *) override;

        void visit(YieldFromStmt *) override;

        void visit(WithStmt *) override;

    public:
        friend std::ostream &operator<<(std::ostream &out, const FormatVisitor &c) {
            return out << c.result;
        }

        using CallbackASTVisitor<std::string, std::string>::transform;

        template<typename T>
        std::string transform(const std::vector<T> &ts) {
            std::vector<std::string> r;
            for (auto &e: ts)
                r.push_back(transform(e));
            return collie::format("{}", collie::join(r, ", "));
        }
    };

} // namespace hercules::ast

template<>
struct collie::formatter<hercules::ast::FormatVisitor> : collie::ostream_formatter {
};
