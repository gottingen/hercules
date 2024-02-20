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
#include <utility>
#include <vector>

#include "hercules/parser/ast.h"
#include "hercules/parser/cache.h"
#include "hercules/parser/common.h"
#include "hercules/parser/ctx.h"
#include "hercules/parser/visitors/visitor.h"

namespace hercules::ast {

    struct json {
        // values={str -> null} -> string value
        // values={i -> json} -> list (if list=true)
        // values={...} -> dictionary
        std::unordered_map<std::string, std::shared_ptr<json>> values;
        bool list;

        json();

        json(const std::string &s);

        json(const std::string &s, const std::string &v);

        json(const std::vector<std::shared_ptr<json>> &vs);

        json(const std::vector<std::string> &vs);

        json(const std::unordered_map<std::string, std::string> &vs);

        std::string toString();

        std::shared_ptr<json> get(const std::string &s);

        std::shared_ptr<json> set(const std::string &s, const std::string &value);

        std::shared_ptr<json> set(const std::string &s, const std::shared_ptr<json> &value);
    };

    struct DocContext;

    struct DocShared {
        int itemID = 1;
        std::shared_ptr<json> j;
        std::unordered_map<std::string, std::shared_ptr<DocContext>> modules;
        std::string argv0;
        Cache *cache = nullptr;
        std::unordered_map<int, std::vector<std::string>> generics;

        DocShared() {}
    };

    struct DocContext : public Context<int> {
        std::shared_ptr<DocShared> shared;

        explicit DocContext(std::shared_ptr<DocShared> shared)
                : Context<int>(""), shared(std::move(shared)) {}

        std::shared_ptr<int> find(const std::string &s) const override;
    };

    struct DocVisitor : public CallbackASTVisitor<std::shared_ptr<json>, std::string> {
        std::shared_ptr<DocContext> ctx;
        std::shared_ptr<json> resultExpr;
        std::string resultStmt;

    public:
        explicit DocVisitor(std::shared_ptr<DocContext> ctx) : ctx(std::move(ctx)) {}

        static std::shared_ptr<json> apply(const std::string &argv0,
                                           const std::vector<std::string> &files);

        std::shared_ptr<json> transform(const ExprPtr &e) override;

        std::string transform(const StmtPtr &e) override;

        void transformModule(StmtPtr stmt);

        std::shared_ptr<json> jsonify(const hercules::SrcInfo &s);

        std::vector<StmtPtr> flatten(StmtPtr stmt, std::string *docstr = nullptr,
                                     bool deep = true);

    public:
        void visit(IntExpr *) override;

        void visit(IdExpr *) override;

        void visit(IndexExpr *) override;

        void visit(FunctionStmt *) override;

        void visit(ClassStmt *) override;

        void visit(AssignStmt *) override;

        void visit(ImportStmt *) override;
    };

} // namespace hercules::ast
