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

#include "hercules/hir/transform/pass.h"
#include "hercules/hir/transform/rewrite.h"

namespace hercules::ir::transform::cleanup {

    /// Canonicalization pass that flattens nested series
    /// flows, puts operands in a predefined order, etc.
    class CanonicalizationPass : public OperatorPass, public Rewriter {
    private:
        std::string sideEffectsKey;

    public:
        /// Constructs a canonicalization pass
        /// @param sideEffectsKey the side effect analysis' key
        CanonicalizationPass(const std::string &sideEffectsKey)
                : OperatorPass(/*childrenFirst=*/true), sideEffectsKey(sideEffectsKey) {}

        static const std::string KEY;

        std::string getKey() const override { return KEY; }

        void run(Module *m) override;

        void handle(CallInstr *) override;

        void handle(SeriesFlow *) override;

    private:
        void registerStandardRules(Module *m);
    };

} // namespace hercules::ir::transform::cleanup
