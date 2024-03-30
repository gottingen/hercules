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

#include <hercules/hir/transform/folding/const_prop.h>
#include <hercules/hir/analyze/dataflow/reaching.h>
#include <hercules/hir/analyze/module/global_vars.h>
#include <hercules/hir/util/cloning.h>

namespace hercules::ir::transform::folding {

    namespace {
        bool okConst(Value *v) {
            return v && (isA<IntConst>(v) || isA<FloatConst>(v) || isA<BoolConst>(v));
        }
    } // namespace

    const std::string ConstPropPass::KEY = "core-folding-const-prop";

    void ConstPropPass::handle(VarValue *v) {
        auto *M = v->getModule();

        auto *var = v->getVar();

        Value *replacement;
        if (var->isGlobal()) {
            auto *r = getAnalysisResult<analyze::module::GlobalVarsResult>(globalVarsKey);
            if (!r)
                return;

            auto it = r->assignments.find(var->getId());
            if (it == r->assignments.end())
                return;

            auto *constDef = M->getValue(it->second);
            if (!okConst(constDef))
                return;

            util::CloneVisitor cv(M);
            replacement = cv.clone(constDef);
        } else {
            auto *r = getAnalysisResult<analyze::dataflow::RDResult>(reachingDefKey);
            if (!r)
                return;
            auto *c = r->cfgResult;

            auto it = r->results.find(getParentFunc()->getId());
            auto it2 = c->graphs.find(getParentFunc()->getId());
            if (it == r->results.end() || it2 == c->graphs.end())
                return;

            auto *rd = it->second.get();
            auto *cfg = it2->second.get();

            auto reaching = rd->getReachingDefinitions(var, v);

            if (reaching.size() != 1)
                return;

            auto def = *reaching.begin();
            if (def == -1)
                return;

            auto *constDef = cfg->getValue(def);
            if (!okConst(constDef))
                return;

            util::CloneVisitor cv(M);
            replacement = cv.clone(constDef);
        }

        v->replaceAll(replacement);
    }

}  // namespace hercules::ir::transform::folding
