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

#ifndef TURBO_PLATFORM_CONFIG_ATTRIBUTE_SANITIZE_H_
#define TURBO_PLATFORM_CONFIG_ATTRIBUTE_SANITIZE_H_

#include "turbo/platform/config/compiler_traits.h"

// TURBO_NO_SANITIZE_ADDRESS
//
// Tells the AddressSanitizer (or other memory testing tools) to ignore a given
// function. Useful for cases when a function reads random locations on stack,
// calls _exit from a cloned subprocess, deliberately accesses buffer
// out of bounds or does other scary things with memory.
// NOTE: GCC supports AddressSanitizer(asan) since 4.8.
// https://gcc.gnu.org/gcc-4.8/changes.html
#if TURBO_HAVE_ATTRIBUTE(no_sanitize_address)
#define TURBO_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#elif defined(_MSC_VER) && _MSC_VER >= 1928
// https://docs.microsoft.com/en-us/cpp/cpp/no-sanitize-address
#define TURBO_NO_SANITIZE_ADDRESS __declspec(no_sanitize_address)
#else
#define TURBO_NO_SANITIZE_ADDRESS
#endif



// TURBO_NO_SANITIZE_MEMORY
//
// Tells the MemorySanitizer to relax the handling of a given function. All "Use
// of uninitialized value" warnings from such functions will be suppressed, and
// all values loaded from memory will be considered fully initialized.  This
// attribute is similar to the TURBO_NO_SANITIZE_ADDRESS attribute
// above, but deals with initialized-ness rather than addressability issues.
// NOTE: MemorySanitizer(msan) is supported by Clang but not GCC.
#if TURBO_HAVE_ATTRIBUTE(no_sanitize_memory)
#define TURBO_NO_SANITIZE_MEMORY __attribute__((no_sanitize_memory))
#else
#define TURBO_NO_SANITIZE_MEMORY
#endif

// TURBO_NO_SANITIZE_THREAD
//
// Tells the ThreadSanitizer to not instrument a given function.
// NOTE: GCC supports ThreadSanitizer(tsan) since 4.8.
// https://gcc.gnu.org/gcc-4.8/changes.html
#if TURBO_HAVE_ATTRIBUTE(no_sanitize_thread)
#define TURBO_NO_SANITIZE_THREAD __attribute__((no_sanitize_thread))
#else
#define TURBO_NO_SANITIZE_THREAD
#endif

// TURBO_NO_SANITIZE_CFI
//
// Tells the ControlFlowIntegrity sanitizer to not instrument a given function.
// See https://clang.llvm.org/docs/ControlFlowIntegrity.html for details.
#if TURBO_HAVE_ATTRIBUTE(no_sanitize)
#define TURBO_NO_SANITIZE_CFI __attribute__((no_sanitize("cfi")))
#else
#define TURBO_NO_SANITIZE_CFI
#endif


// TURBO_NO_SANITIZE_SAFESTACK
//
// Tells the SafeStack to not instrument a given function.
// See https://clang.llvm.org/docs/SafeStack.html for details.
#if TURBO_HAVE_ATTRIBUTE(no_sanitize)
#define TURBO_NO_SANITIZE_SAFESTACK \
  __attribute__((no_sanitize("safe-stack")))
#else
#define TURBO_NO_SANITIZE_SAFESTACK
#endif




#endif  // TURBO_PLATFORM_CONFIG_ATTRIBUTE_SANITIZE_H_
