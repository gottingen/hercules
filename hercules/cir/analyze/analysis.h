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

#include "hercules/cir/module.h"
#include "hercules/cir/transform/pass.h"

namespace hercules {
namespace ir {
namespace analyze {

/// Analysis result base struct.
struct Result {
  virtual ~Result() noexcept = default;
};

/// Base class for IR analyses.
class Analysis {
private:
  transform::PassManager *manager = nullptr;

public:
  virtual ~Analysis() noexcept = default;

  /// @return a unique key for this pass
  virtual std::string getKey() const = 0;

  /// Execute the analysis.
  /// @param module the module
  virtual std::unique_ptr<Result> run(const Module *module) = 0;

  /// Sets the manager.
  /// @param mng the new manager
  void setManager(transform::PassManager *mng) { manager = mng; }
  /// Returns the result of a given analysis.
  /// @param key the analysis key
  template <typename AnalysisType>
  AnalysisType *getAnalysisResult(const std::string &key) {
    return static_cast<AnalysisType *>(doGetAnalysis(key));
  }

private:
  analyze::Result *doGetAnalysis(const std::string &key);
};

} // namespace analyze
} // namespace ir
} // namespace hercules
