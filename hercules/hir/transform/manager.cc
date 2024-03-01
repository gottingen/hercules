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

#include "manager.h"

#include <unordered_set>

#include "hercules/hir/analyze/analysis.h"
#include "hercules/hir/analyze/dataflow/capture.h"
#include "hercules/hir/analyze/dataflow/cfg.h"
#include "hercules/hir/analyze/dataflow/dominator.h"
#include "hercules/hir/analyze/dataflow/reaching.h"
#include "hercules/hir/analyze/module/global_vars.h"
#include "hercules/hir/analyze/module/side_effect.h"
#include "hercules/hir/transform/folding/folding.h"
#include "hercules/hir/transform/lowering/imperative.h"
#include "hercules/hir/transform/lowering/pipeline.h"
#include "hercules/hir/transform/manager.h"
#include "hercules/hir/transform/parallel/openmp.h"
#include "hercules/hir/transform/pass.h"
#include "hercules/hir/transform/pythonic/dict.h"
#include "hercules/hir/transform/pythonic/generator.h"
#include "hercules/hir/transform/pythonic/io.h"
#include "hercules/hir/transform/pythonic/list.h"
#include "hercules/hir/transform/pythonic/str.h"
#include "hercules/util/common.h"

namespace hercules::ir::transform {

    std::string PassManager::KeyManager::getUniqueKey(const std::string &key) {
        // make sure we can't ever produce duplicate "unique'd" keys
        seqassertn(key.find(':') == std::string::npos,
                   "pass key '{}' contains invalid character ':'", key);
        auto it = keys.find(key);
        if (it == keys.end()) {
            keys.emplace(key, 1);
            return key;
        } else {
            auto id = ++(it->second);
            return key + ":" + std::to_string(id);
        }
    }

    std::string PassManager::registerPass(std::unique_ptr<Pass> pass,
                                          const std::string &insertBefore,
                                          std::vector<std::string> reqs,
                                          std::vector<std::string> invalidates) {
        std::string key = pass->getKey();
        if (isDisabled(key))
            return "";
        key = km.getUniqueKey(key);

        for (const auto &req: reqs) {
            seqassertn(deps.find(req) != deps.end(), "required key '{}' not found", req);
            deps[req].push_back(key);
        }

        passes.insert(std::make_pair(
                key, PassMetadata(std::move(pass), std::move(reqs), std::move(invalidates))));
        passes[key].pass->setManager(this);
        if (insertBefore.empty()) {
            executionOrder.push_back(key);
        } else {
            auto it = std::find(executionOrder.begin(), executionOrder.end(), insertBefore);
            seqassertn(it != executionOrder.end(), "pass with key '{}' not found in manager",
                       insertBefore);
            executionOrder.insert(it, key);
        }
        return key;
    }

    std::string PassManager::registerAnalysis(std::unique_ptr<analyze::Analysis> analysis,
                                              std::vector<std::string> reqs) {

        std::string key = analysis->getKey();
        if (isDisabled(key))
            return "";
        key = km.getUniqueKey(key);

        for (const auto &req: reqs) {
            seqassertn(deps.find(req) != deps.end(), "required key '{}' not found", req);
            deps[req].push_back(key);
        }

        analyses.insert(
                std::make_pair(key, AnalysisMetadata(std::move(analysis), std::move(reqs))));
        analyses[key].analysis->setManager(this);
        deps[key] = {};
        return key;
    }

    void PassManager::run(Module *module) {
        for (auto &p: executionOrder) {
            runPass(module, p);
        }
    }

    void PassManager::runPass(Module *module, const std::string &name) {
        auto &meta = passes[name];

        auto run = true;
        auto it = 0;

        while (run) {
            for (auto &dep: meta.reqs) {
                runAnalysis(module, dep);
            }

            Timer timer("  ir pass    : " + meta.pass->getKey());
            meta.pass->run(module);
            timer.log();

            for (auto &inv: meta.invalidates)
                invalidate(inv);

            run = meta.pass->shouldRepeat(++it);
        }
    }

    void PassManager::runAnalysis(Module *module, const std::string &name) {
        if (results.find(name) != results.end())
            return;

        auto &meta = analyses[name];
        for (auto &dep: meta.reqs) {
            runAnalysis(module, dep);
        }

        Timer timer("  ir analysis: " + meta.analysis->getKey());
        results[name] = meta.analysis->run(module);
        timer.log();
    }

    void PassManager::invalidate(const std::string &key) {
        std::unordered_set<std::string> open = {key};

        while (!open.empty()) {
            std::unordered_set<std::string> newOpen;
            for (const auto &k: open) {
                if (results.find(k) != results.end()) {
                    results.erase(k);
                    newOpen.insert(deps[k].begin(), deps[k].end());
                }
            }
            open = std::move(newOpen);
        }
    }

    void PassManager::registerStandardPasses(PassManager::Init init) {
        switch (init) {
            case Init::EMPTY:
                break;
            case Init::DEBUG: {
                registerPass(std::make_unique<lowering::PipelineLowering>());
                registerPass(std::make_unique<lowering::ImperativeForFlowLowering>());
                registerPass(std::make_unique<parallel::OpenMPPass>());
                break;
            }
            case Init::RELEASE:
            case Init::JIT: {
                // Pythonic
                registerPass(std::make_unique<pythonic::DictArithmeticOptimization>());
                registerPass(std::make_unique<pythonic::ListAdditionOptimization>());
                registerPass(std::make_unique<pythonic::StrAdditionOptimization>());
                registerPass(std::make_unique<pythonic::GeneratorArgumentOptimization>());
                registerPass(std::make_unique<pythonic::IOCatOptimization>());

                // lowering
                registerPass(std::make_unique<lowering::PipelineLowering>());
                registerPass(std::make_unique<lowering::ImperativeForFlowLowering>());

                // folding
                auto cfgKey = registerAnalysis(std::make_unique<analyze::dataflow::CFAnalysis>());
                auto rdKey = registerAnalysis(
                        std::make_unique<analyze::dataflow::RDAnalysis>(cfgKey), {cfgKey});
                auto domKey = registerAnalysis(
                        std::make_unique<analyze::dataflow::DominatorAnalysis>(cfgKey), {cfgKey});
                auto capKey = registerAnalysis(
                        std::make_unique<analyze::dataflow::CaptureAnalysis>(rdKey, domKey),
                        {rdKey, domKey});
                auto globalKey =
                        registerAnalysis(std::make_unique<analyze::module::GlobalVarsAnalyses>());
                auto seKey1 =
                        registerAnalysis(std::make_unique<analyze::module::SideEffectAnalysis>(
                                                 capKey,
                                                 /*globalAssignmentHasSideEffects=*/true),
                                         {capKey});
                auto seKey2 =
                        registerAnalysis(std::make_unique<analyze::module::SideEffectAnalysis>(
                                                 capKey,
                                                 /*globalAssignmentHasSideEffects=*/false),
                                         {capKey});
                registerPass(std::make_unique<folding::FoldingPassGroup>(
                                     seKey1, rdKey, globalKey, /*repeat=*/5, /*runGlobalDemoton=*/false,
                                     pyNumerics),
                        /*insertBefore=*/"", {seKey1, rdKey, globalKey},
                             {seKey1, rdKey, cfgKey, globalKey, capKey});

                // parallel
                registerPass(std::make_unique<parallel::OpenMPPass>(), /*insertBefore=*/"", {},
                             {cfgKey, globalKey});

                if (init != Init::JIT) {
                    // Don't demote globals in JIT mode, since they might be used later
                    // by another user input.
                    registerPass(std::make_unique<folding::FoldingPassGroup>(
                                         seKey2, rdKey, globalKey,
                                         /*repeat=*/5,
                                         /*runGlobalDemoton=*/true, pyNumerics),
                            /*insertBefore=*/"", {seKey2, rdKey, globalKey},
                                 {seKey2, rdKey, cfgKey, globalKey});
                }
                break;
            }
            default:
                seqassertn(false, "unknown PassManager init value");
        }
    }

} // namespace hercules::ir::transform
