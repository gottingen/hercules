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

#include "dead_code.h"

#include "hercules/hir/analyze/module/side_effect.h"
#include "hercules/hir/util/cloning.h"

namespace hercules::ir::transform::cleanup {

    namespace {
        BoolConst *boolConst(Value *v) { return cast<BoolConst>(v); }

        IntConst *intConst(Value *v) { return cast<IntConst>(v); }
    } // namespace

    const std::string DeadCodeCleanupPass::KEY = "core-cleanup-dce";

    void DeadCodeCleanupPass::run(Module *m) {
        numReplacements = 0;
        OperatorPass::run(m);
    }

    void DeadCodeCleanupPass::handle(SeriesFlow *v) {
        auto *r = getAnalysisResult<analyze::module::SideEffectResult>(sideEffectsKey);
        auto it = v->begin();
        while (it != v->end()) {
            if (!r->hasSideEffect(*it)) {
                LOG_IR("[{}] no side effect, deleting: {}", KEY, **it);
                numReplacements++;
                it = v->erase(it);
            } else {
                ++it;
            }
        }
    }

    void DeadCodeCleanupPass::handle(IfFlow *v) {
        auto *cond = boolConst(v->getCond());
        if (!cond)
            return;

        auto *M = v->getModule();
        auto condVal = cond->getVal();

        util::CloneVisitor cv(M);
        if (condVal) {
            doReplacement(v, cv.clone(v->getTrueBranch()));
        } else if (auto *f = v->getFalseBranch()) {
            doReplacement(v, cv.clone(f));
        } else {
            doReplacement(v, M->Nr<SeriesFlow>());
        }
    }

    void DeadCodeCleanupPass::handle(WhileFlow *v) {
        auto *cond = boolConst(v->getCond());
        if (!cond)
            return;

        auto *M = v->getModule();
        auto condVal = cond->getVal();
        if (!condVal) {
            doReplacement(v, M->Nr<SeriesFlow>());
        }
    }

    void DeadCodeCleanupPass::handle(ImperativeForFlow *v) {
        auto *start = intConst(v->getStart());
        auto *end = intConst(v->getEnd());
        if (!start || !end)
            return;

        auto stepVal = v->getStep();
        auto startVal = start->getVal();
        auto endVal = end->getVal();

        auto *M = v->getModule();
        if ((stepVal < 0 && startVal <= endVal) || (stepVal > 0 && startVal >= endVal)) {
            doReplacement(v, M->Nr<SeriesFlow>());
        }
    }

    void DeadCodeCleanupPass::handle(TernaryInstr *v) {
        auto *cond = boolConst(v->getCond());
        if (!cond)
            return;

        auto *M = v->getModule();
        auto condVal = cond->getVal();

        util::CloneVisitor cv(M);
        if (condVal) {
            doReplacement(v, cv.clone(v->getTrueValue()));
        } else {
            doReplacement(v, cv.clone(v->getFalseValue()));
        }
    }

    void DeadCodeCleanupPass::doReplacement(Value *og, Value *v) {
        numReplacements++;
        og->replaceAll(v);
    }

} // namespace hercules::ir::transform::cleanup
