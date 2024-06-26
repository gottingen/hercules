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

#include <hercules/hir/transform/pass.h>
#include <hercules/hir/util/visitor.h>

namespace hercules::ir::transform {

    /// Base for rewrite rules.
    class RewriteRule : public util::Visitor {
    private:
        Value *result = nullptr;

    protected:
        void defaultVisit(Node *) override {}

        void setResult(Value *r) { result = r; }

        void resetResult() { setResult(nullptr); }

        Value *getResult() const { return result; }

    public:
        virtual ~RewriteRule() noexcept = default;

        /// Apply the rule.
        /// @param v the value to rewrite
        /// @return nullptr if no rewrite, the replacement otherwise
        Value *apply(Value *v) {
            v->accept(*this);
            auto *replacement = getResult();
            resetResult();
            return replacement;
        }
    };

    /// A collection of rewrite rules.
    class Rewriter {
    private:
        std::unordered_map<std::string, std::unique_ptr<RewriteRule>> rules;
        int numReplacements = 0;

    public:
        /// Adds a given rewrite rule with the given key.
        /// @param key the rule's key
        /// @param rule the rewrite rule
        void registerRule(const std::string &key, std::unique_ptr<RewriteRule> rule) {
            rules.emplace(std::make_pair(key, std::move(rule)));
        }

        /// Applies all rewrite rules to the given node, and replaces the given
        /// node with the result of the rewrites.
        /// @param v the node to rewrite
        void rewrite(Value *v) {
            Value *result = v;
            for (auto &r: rules) {
                if (auto *rep = r.second->apply(result)) {
                    ++numReplacements;
                    result = rep;
                }
            }
            if (v != result)
                v->replaceAll(result);
        }

        /// @return the number of replacements
        int getNumReplacements() const { return numReplacements; }

        /// Sets the replacement count to zero.
        void reset() { numReplacements = 0; }
    };

} // namespace hercules::ir::transform
