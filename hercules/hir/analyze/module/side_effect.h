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

#include <unordered_map>

#include "hercules/hir/analyze/analysis.h"
#include "hercules/hir/util/side_effect.h"

namespace hercules::ir::analyze::module {

    struct SideEffectResult : public Result {
        /// mapping of ID to corresponding node's side effect status
        std::unordered_map<id_t, util::SideEffectStatus> result;

        SideEffectResult(std::unordered_map<id_t, util::SideEffectStatus> result)
                : result(std::move(result)) {}

        /// @param v the value to check
        /// @return true if the node has side effects (false positives allowed)
        bool hasSideEffect(const Value *v) const;
    };

    class SideEffectAnalysis : public Analysis {
    private:
        /// the capture analysis key
        std::string capAnalysisKey;
        /// true if assigning to a global variable automatically has side effects
        bool globalAssignmentHasSideEffects;

    public:
        static const std::string KEY;

        /// Constructs a side effect analysis.
        /// @param globalAssignmentHasSideEffects true if global variable assignment
        /// automatically has side effects
        explicit SideEffectAnalysis(const std::string &capAnalysisKey,
                                    bool globalAssignmentHasSideEffects = true)
                : Analysis(), capAnalysisKey(capAnalysisKey),
                  globalAssignmentHasSideEffects(globalAssignmentHasSideEffects) {}

        std::string getKey() const override { return KEY; }

        std::unique_ptr<Result> run(const Module *m) override;
    };

} // namespace hercules::ir::analyze::module
