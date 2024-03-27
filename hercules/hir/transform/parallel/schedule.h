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

#include "hercules/hir/value.h"

namespace hercules {
namespace ir {

class Value;

namespace transform {
namespace parallel {

struct OMPSched {
  int code;
  bool dynamic;
  Value *threads;
  Value *chunk;
  bool ordered;
  int64_t collapse;
  bool gpu;

  explicit OMPSched(int code = -1, bool dynamic = false, Value *threads = nullptr,
                    Value *chunk = nullptr, bool ordered = false, int64_t collapse = 0,
                    bool gpu = false);
  explicit OMPSched(const std::string &code, Value *threads = nullptr,
                    Value *chunk = nullptr, bool ordered = false, int64_t collapse = 0,
                    bool gpu = false);
  OMPSched(const OMPSched &s)
      : code(s.code), dynamic(s.dynamic), threads(s.threads), chunk(s.chunk),
        ordered(s.ordered), collapse(s.collapse), gpu(s.gpu) {}

  std::vector<Value *> getUsedValues() const;
  int replaceUsedValue(id_t id, Value *newValue);
};

} // namespace parallel
} // namespace transform
} // namespace ir
} // namespace hercules
