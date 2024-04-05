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

#include <any>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <hercules/parser/peg/peglib.h>
#include <hercules/parser/ast.h>
#include <hercules/parser/cache.h>
#include <hercules/parser/common.h>

namespace hercules::ast {

    struct ParseContext {
        Cache *cache;
        std::stack<int> indent;
        int parens;
        int line_offset, col_offset;

        ParseContext(Cache *cache, int parens = 0, int line_offset = 0, int col_offset = 0)
                : cache(cache), parens(parens), line_offset(line_offset), col_offset(col_offset) {
        }

        bool hasCustomStmtKeyword(const std::string &kwd, bool hasExpr) const {
            auto i = cache->customBlockStmts.find(kwd);
            if (i != cache->customBlockStmts.end())
                return i->second.first == hasExpr;
            return false;
        }

        bool hasCustomExprStmt(const std::string &kwd) const {
            return in(cache->customExprStmts, kwd);
        }
    };

} // namespace hercules::ast

void init_hercules_rules(peg::Grammar &);

void init_hercules_actions(peg::Grammar &);

void init_omp_rules(peg::Grammar &);

void init_omp_actions(peg::Grammar &);
