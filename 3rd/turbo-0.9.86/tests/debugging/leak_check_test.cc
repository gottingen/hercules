// Copyright 2020 The Turbo Authors.
//
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

#include <string>

#include "turbo/base/internal/raw_logging.h"
#include "turbo/debugging/leak_check.h"
#include "turbo/platform/port.h"
#include "gtest/gtest.h"

namespace {

TEST(LeakCheckTest, IgnoreLeakSuppressesLeakedMemoryErrors) {
  if (!turbo::LeakCheckerIsActive()) {
    GTEST_SKIP() << "LeakChecker is not active";
  }
  auto foo = turbo::IgnoreLeak(new std::string("some ignored leaked string"));
  TURBO_RAW_LOG(INFO, "Ignoring leaked string %s", foo->c_str());
}

TEST(LeakCheckTest, LeakCheckDisablerIgnoresLeak) {
  if (!turbo::LeakCheckerIsActive()) {
    GTEST_SKIP() << "LeakChecker is not active";
  }
  turbo::LeakCheckDisabler disabler;
  auto foo = new std::string("some string leaked while checks are disabled");
  TURBO_RAW_LOG(INFO, "Ignoring leaked string %s", foo->c_str());
}

}  // namespace
