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
//
// This file is an Akaros-specific part of spinlock_wait.cc

#include <atomic>

#include "turbo/platform/internal/scheduling_mode.h"

extern "C" {

TURBO_WEAK void turbo_internal_spin_lock_delay(
    std::atomic<uint32_t>* /* lock_word */, uint32_t /* value */,
    int /* loop */, turbo::base_internal::SchedulingMode /* mode */) {
  // In Akaros, one must take care not to call anything that could cause a
  // malloc(), a blocking system call, or a uthread_yield() while holding a
  // spinlock. Our callers assume will not call into libraries or other
  // arbitrary code.
}

TURBO_WEAK void turbo_internal_spin_lock_wake(
    std::atomic<uint32_t>* /* lock_word */, bool /* all */) {}

}  // extern "C"
