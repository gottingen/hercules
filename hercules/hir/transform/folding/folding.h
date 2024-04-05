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

#include <hercules/hir/transform/cleanup/canonical.h>
#include <hercules/hir/transform/cleanup/dead_code.h>
#include <hercules/hir/transform/cleanup/global_demote.h>
#include <hercules/hir/transform/pass.h>

namespace hercules::ir::transform::folding {

    class FoldingPass;

    /// Group of constant folding passes.
    class FoldingPassGroup : public PassGroup {
    private:
        cleanup::GlobalDemotionPass *gd;
        cleanup::CanonicalizationPass *canon;
        FoldingPass *fp;
        cleanup::DeadCodeCleanupPass *dce;

    public:
        static const std::string KEY;

        std::string getKey() const override { return KEY; }

        /// @param sideEffectsPass the key of the side effects pass
        /// @param reachingDefPass the key of the reaching definitions pass
        /// @param globalVarPass the key of the global variables pass
        /// @param repeat default number of times to repeat the pass
        /// @param runGlobalDemotion whether to demote globals if possible
        /// @param pyNumerics whether to use Python (vs. C) semantics when folding
        FoldingPassGroup(const std::string &sideEffectsPass,
                         const std::string &reachingDefPass, const std::string &globalVarPass,
                         int repeat = 5, bool runGlobalDemotion = true,
                         bool pyNumerics = false);

        bool shouldRepeat(int num) const override;
    };

} // namespace hercules::ir::transform::folding
