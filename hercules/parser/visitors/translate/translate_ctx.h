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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hercules/hir/cir.h"
#include "hercules/hir/types/types.h"
#include "hercules/parser/cache.h"
#include "hercules/parser/common.h"
#include "hercules/parser/ctx.h"

namespace hercules::ast {

    /**
     * IR context object description.
     * This represents an identifier that can be either a function, a class (type), or a
     * variable.
     */
    struct TranslateItem {
        enum Kind {
            Func, Type, Var
        } kind;
        /// IR handle.
        union {
            hercules::ir::Var *var;
            hercules::ir::Func *func;
            hercules::ir::types::Type *type;
        } handle;
        /// Base function pointer.
        hercules::ir::BodiedFunc *base;

        TranslateItem(Kind k, hercules::ir::BodiedFunc *base)
                : kind(k), handle{nullptr}, base(base) {}

        const hercules::ir::BodiedFunc *getBase() const { return base; }

        hercules::ir::Func *getFunc() const { return kind == Func ? handle.func : nullptr; }

        hercules::ir::types::Type *getType() const {
            return kind == Type ? handle.type : nullptr;
        }

        hercules::ir::Var *getVar() const { return kind == Var ? handle.var : nullptr; }
    };

    /**
     * A variable table (context) for the IR translation stage.
     */
    struct TranslateContext : public Context<TranslateItem> {
        /// A pointer to the shared cache.
        Cache *cache;
        /// Stack of function bases.
        std::vector<hercules::ir::BodiedFunc *> bases;
        /// Stack of IR series (blocks).
        std::vector<hercules::ir::SeriesFlow *> series;
        /// Stack of sequence items for attribute initialization.
        std::vector<std::vector<std::pair<ExprAttr, ir::Value *>>> seqItems;

    public:
        TranslateContext(Cache *cache);

        using Context<TranslateItem>::add;

        /// Convenience method for adding an object to the context.
        std::shared_ptr<TranslateItem> add(TranslateItem::Kind kind, const std::string &name,
                                           void *type);

        std::shared_ptr<TranslateItem> find(const std::string &name) const override;

        std::shared_ptr<TranslateItem> forceFind(const std::string &name) const;

        /// Convenience method for adding a series.
        void addSeries(hercules::ir::SeriesFlow *s);

        void popSeries();

    public:
        hercules::ir::Module *getModule() const;

        hercules::ir::BodiedFunc *getBase() const;

        hercules::ir::SeriesFlow *getSeries() const;
    };

} // namespace hercules::ast
