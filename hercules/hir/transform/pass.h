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

#include "hercules/hir/module.h"
#include "hercules/hir/util/operator.h"

namespace hercules::ir::analyze {
    struct Result;
}  // hercules::ir::analyze

namespace hercules::ir::transform {

    class PassManager;

    /// General pass base class.
    class Pass {
    private:
        PassManager *manager = nullptr;

    public:
        virtual ~Pass() noexcept = default;

        /// @return a unique key for this pass
        virtual std::string getKey() const = 0;

        /// Execute the pass.
        /// @param module the module
        virtual void run(Module *module) = 0;

        /// Determine if pass should repeat.
        /// @param num how many times this pass has already run
        /// @return true if pass should repeat
        virtual bool shouldRepeat(int num) const { return false; }

        /// Sets the manager.
        /// @param mng the new manager
        virtual void setManager(PassManager *mng) { manager = mng; }

        /// Returns the result of a given analysis.
        /// @param key the analysis key
        /// @return the analysis result
        template<typename AnalysisType>
        AnalysisType *getAnalysisResult(const std::string &key) {
            return static_cast<AnalysisType *>(doGetAnalysis(key));
        }

    private:
        analyze::Result *doGetAnalysis(const std::string &key);
    };

    class PassGroup : public Pass {
    private:
        int repeat;
        std::vector<std::unique_ptr<Pass>> passes;

    public:
        explicit PassGroup(int repeat = 0, std::vector<std::unique_ptr<Pass>> passes = {})
                : Pass(), repeat(repeat), passes(std::move(passes)) {}

        virtual ~PassGroup() noexcept = default;

        void push_back(std::unique_ptr<Pass> p) { passes.push_back(std::move(p)); }

        /// @return default number of times pass should repeat
        int getRepeat() const { return repeat; }

        /// Sets the default number of times pass should repeat.
        /// @param r number of repeats
        void setRepeat(int r) { repeat = r; }

        bool shouldRepeat(int num) const override { return num < repeat; }

        void run(Module *module) override;

        void setManager(PassManager *mng) override;
    };

    /// Pass that runs a single Operator.
    class OperatorPass : public Pass, public util::Operator {
    public:
        /// Constructs an operator pass.
        /// @param childrenFirst true if children should be iterated first
        explicit OperatorPass(bool childrenFirst = false) : util::Operator(childrenFirst) {}

        void run(Module *module) override {
            reset();
            process(module);
        }
    };

} // namespace hercules::ir::transform
