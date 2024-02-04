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

#include "side_effect.h"

namespace hercules {
namespace ir {
namespace util {

const std::string NON_PURE_ATTR = "std.internal.attributes.nonpure";
const std::string PURE_ATTR = "std.internal.attributes.pure";
const std::string NO_SIDE_EFFECT_ATTR = "std.internal.attributes.no_side_effect";
const std::string NO_CAPTURE_ATTR = "std.internal.attributes.nocapture";
const std::string DERIVES_ATTR = "std.internal.attributes.derives";
const std::string SELF_CAPTURES_ATTR = "std.internal.attributes.self_captures";

} // namespace util
} // namespace ir
} // namespace hercules
