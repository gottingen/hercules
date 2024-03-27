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

namespace hercules::ir::transform::cleanup {

    /// Cleanup pass that removes dead code.
    class DeadCodeCleanupPass : public OperatorPass {
    private:
        std::string sideEffectsKey;
        int numReplacements;

    public:
        static const std::string KEY;

        /// Constructs a dead code elimination pass
        /// @param sideEffectsKey the side effect analysis' key
        DeadCodeCleanupPass(std::string sideEffectsKey)
                : OperatorPass(), sideEffectsKey(std::move(sideEffectsKey)), numReplacements(0) {}

        std::string getKey() const override { return KEY; }

        void run(Module *m) override;

        void handle(SeriesFlow *v) override;

        void handle(IfFlow *v) override;

        void handle(WhileFlow *v) override;

        void handle(ImperativeForFlow *v) override;

        void handle(TernaryInstr *v) override;

        /// @return the number of replacements
        int getNumReplacements() const { return numReplacements; }

    private:
        void doReplacement(Value *og, Value *v);
    };

} // namespace hercules::ir::transform::cleanup
