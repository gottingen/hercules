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

#include "folding.h"

#include "hercules/cir/transform/folding/const_fold.h"
#include "hercules/cir/transform/folding/const_prop.h"

namespace hercules::ir::transform::folding {


    const std::string FoldingPassGroup::KEY = "core-folding-pass-group";

    FoldingPassGroup::FoldingPassGroup(const std::string &sideEffectsPass,
                                       const std::string &reachingDefPass,
                                       const std::string &globalVarPass, int repeat,
                                       bool runGlobalDemotion, bool pyNumerics)
            : PassGroup(repeat) {
        auto gdUnique = runGlobalDemotion ? std::make_unique<cleanup::GlobalDemotionPass>()
                                          : std::unique_ptr<cleanup::GlobalDemotionPass>();
        auto canonUnique = std::make_unique<cleanup::CanonicalizationPass>(sideEffectsPass);
        auto fpUnique = std::make_unique<FoldingPass>(pyNumerics);
        auto dceUnique = std::make_unique<cleanup::DeadCodeCleanupPass>(sideEffectsPass);

        gd = gdUnique.get();
        canon = canonUnique.get();
        fp = fpUnique.get();
        dce = dceUnique.get();

        if (runGlobalDemotion)
            push_back(std::move(gdUnique));
        push_back(std::make_unique<ConstPropPass>(reachingDefPass, globalVarPass));
        push_back(std::move(canonUnique));
        push_back(std::move(fpUnique));
        push_back(std::move(dceUnique));
    }

    bool FoldingPassGroup::shouldRepeat(int num) const {
        return PassGroup::shouldRepeat(num) &&
               ((gd && gd->getNumDemotions() != 0) || canon->getNumReplacements() != 0 ||
                fp->getNumReplacements() != 0 || dce->getNumReplacements() != 0);
    }


} // namespace hercules::ir::transform::folding
