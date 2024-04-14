// Copyright 2023 The Turbo Authors.
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

#include "turbo/debugging/stacktrace.h"

#include "turbo/platform/port.h"
#include "gtest/gtest.h"

namespace {

// This test is currently only known to pass on linux/x86_64.
#if defined(__linux__) && defined(__x86_64__)
TURBO_NO_INLINE void Unwind(void* p) {
  TURBO_MAYBE_UNUSED static void* volatile sink = p;
  constexpr int kSize = 16;
  void* stack[kSize];
  int frames[kSize];
  turbo::GetStackTrace(stack, kSize, 0);
  turbo::GetStackFrames(stack, frames, kSize, 0);
}

TURBO_NO_INLINE void HugeFrame() {
  char buffer[1 << 20];
  Unwind(buffer);
  TURBO_BLOCK_TAIL_CALL_OPTIMIZATION();
}

TEST(StackTrace, HugeFrame) {
  // Ensure that the unwinder is not confused by very large stack frames.
  HugeFrame();
  TURBO_BLOCK_TAIL_CALL_OPTIMIZATION();
}
#endif

}  // namespace
