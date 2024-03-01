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

#include "pipeline.h"

#include <algorithm>

#include "hercules/hir/util/cloning.h"
#include "hercules/hir/util/irtools.h"
#include "hercules/hir/util/matching.h"

namespace hercules::ir::transform::lowering {

    namespace {
        Value *callStage(Module *M, PipelineFlow::Stage *stage, Value *last) {
            std::vector<Value *> args;
            for (auto *arg: *stage) {
                args.push_back(arg ? arg : last);
            }
            return M->N<CallInstr>(stage->getCallee()->getSrcInfo(), stage->getCallee(), args);
        }

        Value *convertPipelineToForLoopsHelper(Module *M, BodiedFunc *parent,
                                               const std::vector<PipelineFlow::Stage *> &stages,
                                               unsigned idx = 0, Value *last = nullptr) {
            if (idx >= stages.size())
                return last;

            auto *stage = stages[idx];
            if (idx == 0)
                return convertPipelineToForLoopsHelper(M, parent, stages, idx + 1,
                                                       stage->getCallee());

            auto *prev = stages[idx - 1];
            if (prev->isGenerator()) {
                auto *var = M->Nr<Var>(prev->getOutputElementType());
                parent->push_back(var);
                auto *body = convertPipelineToForLoopsHelper(
                        M, parent, stages, idx + 1, callStage(M, stage, M->Nr<VarValue>(var)));
                auto *loop = M->N<ForFlow>(last->getSrcInfo(), last, util::series(body), var);
                if (stage->isParallel())
                    loop->setParallel();
                return loop;
            } else {
                return convertPipelineToForLoopsHelper(M, parent, stages, idx + 1,
                                                       callStage(M, stage, last));
            }
        }

        Value *convertPipelineToForLoops(PipelineFlow *p, BodiedFunc *parent) {
            std::vector<PipelineFlow::Stage *> stages;
            for (auto &stage: *p) {
                stages.push_back(&stage);
            }
            return convertPipelineToForLoopsHelper(p->getModule(), parent, stages);
        }
    } // namespace

    const std::string PipelineLowering::KEY = "core-pipeline-lowering";

    void PipelineLowering::handle(PipelineFlow *v) {
        v->replaceAll(convertPipelineToForLoops(v, cast<BodiedFunc>(getParentFunc())));
    }

} // namespace hercules::ir::transform::lowering
