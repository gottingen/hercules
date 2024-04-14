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
// This file is intended solely for spinlock.h.
// It provides ThreadSanitizer annotations for custom mutexes.
// See <sanitizer/tsan_interface.h> for meaning of these annotations.

#ifndef TURBO_PLATFORM_INTERNAL_TSAN_MUTEX_INTERFACE_H_
#define TURBO_PLATFORM_INTERNAL_TSAN_MUTEX_INTERFACE_H_

#include "turbo/platform/port.h"

// TURBO_INTERNAL_HAVE_TSAN_INTERFACE
// Macro intended only for internal use.
//
// Checks whether LLVM Thread Sanitizer interfaces are available.
// First made available in LLVM 5.0 (Sep 2017).
#ifdef TURBO_INTERNAL_HAVE_TSAN_INTERFACE
#error "TURBO_INTERNAL_HAVE_TSAN_INTERFACE cannot be directly set."
#endif

#if defined(TURBO_HAVE_THREAD_SANITIZER) && defined(__has_include)
#if __has_include(<sanitizer/tsan_interface.h>)
#define TURBO_INTERNAL_HAVE_TSAN_INTERFACE 1
#endif
#endif

#ifdef TURBO_INTERNAL_HAVE_TSAN_INTERFACE
#include <sanitizer/tsan_interface.h>

#define TURBO_TSAN_MUTEX_CREATE __tsan_mutex_create
#define TURBO_TSAN_MUTEX_DESTROY __tsan_mutex_destroy
#define TURBO_TSAN_MUTEX_PRE_LOCK __tsan_mutex_pre_lock
#define TURBO_TSAN_MUTEX_POST_LOCK __tsan_mutex_post_lock
#define TURBO_TSAN_MUTEX_PRE_UNLOCK __tsan_mutex_pre_unlock
#define TURBO_TSAN_MUTEX_POST_UNLOCK __tsan_mutex_post_unlock
#define TURBO_TSAN_MUTEX_PRE_SIGNAL __tsan_mutex_pre_signal
#define TURBO_TSAN_MUTEX_POST_SIGNAL __tsan_mutex_post_signal
#define TURBO_TSAN_MUTEX_PRE_DIVERT __tsan_mutex_pre_divert
#define TURBO_TSAN_MUTEX_POST_DIVERT __tsan_mutex_post_divert

#else

#define TURBO_TSAN_MUTEX_CREATE(...)
#define TURBO_TSAN_MUTEX_DESTROY(...)
#define TURBO_TSAN_MUTEX_PRE_LOCK(...)
#define TURBO_TSAN_MUTEX_POST_LOCK(...)
#define TURBO_TSAN_MUTEX_PRE_UNLOCK(...)
#define TURBO_TSAN_MUTEX_POST_UNLOCK(...)
#define TURBO_TSAN_MUTEX_PRE_SIGNAL(...)
#define TURBO_TSAN_MUTEX_POST_SIGNAL(...)
#define TURBO_TSAN_MUTEX_PRE_DIVERT(...)
#define TURBO_TSAN_MUTEX_POST_DIVERT(...)

#endif

#endif  // TURBO_PLATFORM_INTERNAL_TSAN_MUTEX_INTERFACE_H_
