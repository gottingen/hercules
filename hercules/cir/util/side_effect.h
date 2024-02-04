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

#include <string>

namespace hercules {
namespace ir {
namespace util {

/// Function side effect status. "Pure" functions by definition give the same
/// output for the same inputs and have no side effects. "No side effect"
/// functions have no side effects, but can give different outputs for the
/// same input (e.g. time() is one such function). "No capture" functions do
/// not capture any of their arguments; note that capturing an argument is
/// considered a side effect. Therefore, we have pure < no_side_effect <
/// no_capture < unknown, where "<" denotes subset. The enum values are also
/// ordered in this way, which is relied on by the implementation.
enum SideEffectStatus {
  PURE = 0,
  NO_SIDE_EFFECT,
  NO_CAPTURE,
  UNKNOWN,
};

extern const std::string NON_PURE_ATTR;
extern const std::string PURE_ATTR;
extern const std::string NO_SIDE_EFFECT_ATTR;
extern const std::string NO_CAPTURE_ATTR;
extern const std::string DERIVES_ATTR;
extern const std::string SELF_CAPTURES_ATTR;

} // namespace util
} // namespace ir
} // namespace hercules
