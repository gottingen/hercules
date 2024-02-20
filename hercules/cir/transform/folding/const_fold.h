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

#include <memory>
#include <unordered_map>

#include "hercules/cir/transform/folding/rule.h"
#include "hercules/cir/transform/pass.h"

namespace hercules::ir::transform::folding {

    class FoldingPass : public OperatorPass, public Rewriter {
    private:
        bool pyNumerics;

        void registerStandardRules(Module *m);

    public:
        /// Constructs a folding pass.
        FoldingPass(bool pyNumerics = false)
                : OperatorPass(/*childrenFirst=*/true), pyNumerics(pyNumerics) {}

        static const std::string KEY;

        std::string getKey() const override { return KEY; }

        void run(Module *m) override;

        void handle(CallInstr *v) override;
    };

} // namespace hercules::ir::transform::folding
