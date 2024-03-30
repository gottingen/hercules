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

#include <memory>
#include <unordered_map>
#include <vector>

#include <hercules/hir/analyze/analysis.h>
#include <hercules/hir/analyze/dataflow/dominator.h>
#include <hercules/hir/analyze/dataflow/reaching.h>
#include <hercules/hir/ir.h>

namespace hercules::ir::analyze::dataflow {


    /// Information about how a function argument is captured.
    struct CaptureInfo {
        /// vector of other argument indices capturing this one
        std::vector<unsigned> argCaptures;
        /// true if the return value of the function captures this argument
        bool returnCaptures = false;
        /// true if this argument is externally captured e.g. by assignment to global
        bool externCaptures = false;
        /// true if this argument is modified
        bool modified = false;

        /// @return true if anything captures
        operator bool() const {
            return !argCaptures.empty() || returnCaptures || externCaptures;
        }

        /// Returns an instance denoting no captures.
        /// @return an instance denoting no captures
        static CaptureInfo nothing() { return {}; }

        /// Returns an instance denoting unknown capture status.
        /// @param func the function containing this argument
        /// @param type the argument's type
        /// @return an instance denoting unknown capture status
        static CaptureInfo unknown(const Func *func, types::Type *type);
    };

    /// Capture analysis result.
    struct CaptureResult : public Result {
        /// the corresponding reaching definitions result
        RDResult *rdResult = nullptr;

        /// the corresponding dominator result
        DominatorResult *domResult = nullptr;

        /// map from function id to capture information, where
        /// each element of the value vector corresponds to an
        /// argument of the function
        std::unordered_map<id_t, std::vector<CaptureInfo>> results;
    };

    /// Capture analysis that runs on all functions.
    class CaptureAnalysis : public Analysis {
    private:
        /// the reaching definitions analysis key
        std::string rdAnalysisKey;
        /// the dominator analysis key
        std::string domAnalysisKey;

    public:
        static const std::string KEY;

        std::string getKey() const override { return KEY; }

        /// Initializes a capture analysis.
        /// @param rdAnalysisKey the reaching definitions analysis key
        /// @param domAnalysisKey the dominator analysis key
        explicit CaptureAnalysis(std::string rdAnalysisKey, std::string domAnalysisKey)
                : rdAnalysisKey(std::move(rdAnalysisKey)),
                  domAnalysisKey(std::move(domAnalysisKey)) {}

        std::unique_ptr<Result> run(const Module *m) override;
    };

    CaptureInfo escapes(const BodiedFunc *func, const Value *value, CaptureResult *cr);

} // namespace hercules::ir::analyze::dataflow
