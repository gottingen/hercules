// Copyright 2021 The Turbo Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "turbo/strings/internal/cordz_update_scope.h"

#include "turbo/platform/port.h"
#include "tests/strings/cordz_test_helpers.h"
#include "turbo/strings/internal/cord_rep_flat.h"
#include "turbo/strings/internal/cordz_info.h"
#include "turbo/strings/internal/cordz_update_tracker.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace turbo {
namespace cord_internal {

namespace {

// Used test values
auto constexpr kTrackCordMethod = CordzUpdateTracker::kConstructorString;

TEST(CordzUpdateScopeTest, ScopeNullptr) {
  CordzUpdateScope scope(nullptr, kTrackCordMethod);
}

TEST(CordzUpdateScopeTest, ScopeSampledCord) {
  TestCordData cord;
  CordzInfo::TrackCord(cord.data, kTrackCordMethod);
  CordzUpdateScope scope(cord.data.cordz_info(), kTrackCordMethod);
  cord.data.cordz_info()->SetCordRep(nullptr);
}

}  // namespace
}  // namespace cord_internal

}  // namespace turbo
