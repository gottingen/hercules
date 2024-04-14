// Copyright 2018 The Turbo Authors.
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

#include "turbo/debugging/symbolize.h"

#ifdef _WIN32
#include <winapifamily.h>
#if !(WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP)) || \
    WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
// UWP doesn't have access to win32 APIs.
#define TURBO_INTERNAL_HAVE_SYMBOLIZE_WIN32
#endif
#endif

// Emscripten symbolization relies on JS. Do not use them in standalone mode.
#if defined(__EMSCRIPTEN__) && !defined(STANDALONE_WASM)
#define TURBO_INTERNAL_HAVE_SYMBOLIZE_WASM
#endif

#if defined(TURBO_INTERNAL_HAVE_ELF_SYMBOLIZE)
#include "turbo/debugging/symbolize_elf.h"
#elif defined(TURBO_INTERNAL_HAVE_SYMBOLIZE_WIN32)
// The Windows Symbolizer only works if PDB files containing the debug info
// are available to the program at runtime.
#include "turbo/debugging/symbolize_win32.h"
#elif defined(__APPLE__)
#include "turbo/debugging/symbolize_darwin.h"
#elif defined(TURBO_INTERNAL_HAVE_SYMBOLIZE_WASM)
#include "turbo/debugging/symbolize_emscripten.h"
#else
#include "turbo/debugging/symbolize_unimplemented.h"
#endif
