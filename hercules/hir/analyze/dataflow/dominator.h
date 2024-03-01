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

#include <set>
#include <unordered_map>
#include <utility>

#include "hercules/hir/analyze/analysis.h"
#include "hercules/hir/analyze/dataflow/cfg.h"

namespace hercules::ir::analyze::dataflow {

    /// Helper to query the dominators of a particular function.
    class DominatorInspector {
    private:
        std::unordered_map<id_t, std::set<id_t>> sets;
        CFGraph *cfg;

    public:
        explicit DominatorInspector(CFGraph *cfg) : cfg(cfg) {}

        /// Do the analysis.
        void analyze();

        /// Checks if one value dominates another.
        /// @param v the value
        /// @param dominator the dominator value
        bool isDominated(const Value *v, const Value *dominator);
    };

    /// Result of a dominator analysis.
    struct DominatorResult : public Result {
        /// the corresponding control flow result
        const CFResult *cfgResult;
        /// the dominator inspectors
        std::unordered_map<id_t, std::unique_ptr<DominatorInspector>> results;

        explicit DominatorResult(const CFResult *cfgResult) : cfgResult(cfgResult) {}
    };

    /// Dominator analysis. Must have control flow-graph available.
    class DominatorAnalysis : public Analysis {
    private:
        /// the control-flow analysis key
        std::string cfAnalysisKey;

    public:
        static const std::string KEY;

        /// Initializes a dominator analysis.
        /// @param cfAnalysisKey the control-flow analysis key
        explicit DominatorAnalysis(std::string cfAnalysisKey)
                : cfAnalysisKey(std::move(cfAnalysisKey)) {}

        std::string getKey() const override { return KEY; }

        std::unique_ptr<Result> run(const Module *m) override;
    };

}  // namespace hercules::ir::analyze::dataflow
