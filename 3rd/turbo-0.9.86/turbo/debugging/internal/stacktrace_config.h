/*
 * Copyright 2020 The Turbo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 * Defines TURBO_STACKTRACE_INL_HEADER to the *-inl.h containing
 * actual unwinder implementation.
 * This header is "private" to stacktrace.cc.
 * DO NOT include it into any other files.
*/
#ifndef TURBO_DEBUGGING_INTERNAL_STACKTRACE_CONFIG_H_
#define TURBO_DEBUGGING_INTERNAL_STACKTRACE_CONFIG_H_

#include "turbo/platform/port.h"

#if defined(TURBO_STACKTRACE_INL_HEADER)
#error TURBO_STACKTRACE_INL_HEADER cannot be directly set

#elif defined(_WIN32)
#define TURBO_STACKTRACE_INL_HEADER \
    "turbo/debugging/internal/stacktrace_win32-inl.h"

#elif defined(__APPLE__)
#ifdef TURBO_HAVE_THREAD_LOCAL
// Thread local support required for UnwindImpl.
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_generic-inl.h"
#endif  // defined(TURBO_HAVE_THREAD_LOCAL)

// Emscripten stacktraces rely on JS. Do not use them in standalone mode.
#elif defined(__EMSCRIPTEN__) && !defined(STANDALONE_WASM)
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_emscripten-inl.h"

#elif defined(__linux__) && !defined(__ANDROID__)

#if defined(NO_FRAME_POINTER) && \
    (defined(__i386__) || defined(__x86_64__) || defined(__aarch64__))
// Note: The libunwind-based implementation is not available to open-source
// users.
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_libunwind-inl.h"
#define STACKTRACE_USES_LIBUNWIND 1
#elif defined(NO_FRAME_POINTER) && defined(__has_include)
#if __has_include(<execinfo.h>)
// Note: When using glibc this may require -funwind-tables to function properly.
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_generic-inl.h"
#endif  // __has_include(<execinfo.h>)
#elif defined(__i386__) || defined(__x86_64__)
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_x86-inl.h"
#elif defined(__ppc__) || defined(__PPC__)
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_powerpc-inl.h"
#elif defined(__aarch64__)
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_aarch64-inl.h"
#elif defined(__riscv)
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_riscv-inl.h"
#elif defined(__has_include)
#if __has_include(<execinfo.h>)
// Note: When using glibc this may require -funwind-tables to function properly.
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_generic-inl.h"
#endif  // __has_include(<execinfo.h>)
#endif  // defined(__has_include)

#endif  // defined(__linux__) && !defined(__ANDROID__)

// Fallback to the empty implementation.
#if !defined(TURBO_STACKTRACE_INL_HEADER)
#define TURBO_STACKTRACE_INL_HEADER \
  "turbo/debugging/internal/stacktrace_unimplemented-inl.h"
#endif

#endif  // TURBO_DEBUGGING_INTERNAL_STACKTRACE_CONFIG_H_
