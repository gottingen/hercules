// Copyright 2022 ByteDance Ltd. and/or its affiliates.
// Taken from https://github.com/abseil/abseil-cpp/blob/master/absl/hash/internal/wyhash.h
//
// Copyright 2020 The Abseil Authors
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
//
// This file provides the Google-internal implementation of the Wyhash
// algorithm.
//
// Wyhash is a fast hash function for hash tables, the fastest we've currently
// (late 2020) found that passes the SMHasher tests. The algorithm relies on
// intrinsic 128-bit multiplication for speed. This is not meant to be secure -
// just fast.

#pragma once

#include <stdint.h>
#include <stdlib.h>

#include <hercules/runtime/hash/base/config.h>

namespace matxscript {
namespace runtime {
namespace hash_internal {

// Hash function for a byte array. A 64-bit seed and a set of five 64-bit
// integers are hashed into the result.
//
// To allow all hashable types (including string_view and Span) to depend on
// this algorithm, we keep the API low-level, with as few dependencies as
// possible.
#ifdef MATXSCRIPT_HAVE_INTRINSIC_INT128
typedef __uint128_t uint128;

uint64_t Wyhash(const void* data, size_t len, uint64_t seed, const uint64_t salt[5]) noexcept;
#endif  // MATXSCRIPT_HAVE_INTRINSIC_INT128

}  // namespace hash_internal
}  // namespace runtime
}  // namespace matxscript
