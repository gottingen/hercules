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

#include "hercules/cir/transform/pass.h"

namespace hercules::ir::transform::cleanup {

    /// Demotes global variables that are used in only one
    /// function to locals of that function.
    class GlobalDemotionPass : public Pass {
    private:
        /// number of variables we've demoted
        int numDemotions;

    public:
        static const std::string KEY;

        /// Constructs a global variable demotion pass
        GlobalDemotionPass() : Pass(), numDemotions(0) {}

        std::string getKey() const override { return KEY; }

        void run(Module *v) override;

        /// @return number of variables we've demoted
        int getNumDemotions() const { return numDemotions; }
    };

} // namespace hercules::ir::transform::cleanup
