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

#include <hercules/hir/transform/pass.h>

namespace hercules {
namespace ir {
namespace transform {
namespace parallel {

class OpenMPPass : public OperatorPass {
public:
  /// Constructs an OpenMP pass.
  OpenMPPass() : OperatorPass(/*childrenFirst=*/true) {}

  static const std::string KEY;
  std::string getKey() const override { return KEY; }

  void handle(ForFlow *) override;
  void handle(ImperativeForFlow *) override;
};

} // namespace parallel
} // namespace transform
} // namespace ir
} // namespace hercules
