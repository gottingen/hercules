// Copyright 2020 The Turbo Authors
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
// This file provides the Google-internal implementation of LowLevelHash.
//
// LowLevelHash is a fast hash function for hash tables, the fastest we've
// currently (late 2020) found that passes the SMHasher tests. The algorithm
// relies on intrinsic 128-bit multiplication for speed. This is not meant to be
// secure - just fast.
//
// It is closely based on a version of wyhash, but does not maintain or
// guarantee future compatibility with it.

#ifndef TURBO_HASH_BYTES_LOW_LEVEL_HASH_H_
#define TURBO_HASH_BYTES_LOW_LEVEL_HASH_H_

#include <stdint.h>
#include <stdlib.h>

#include "turbo/platform/port.h"
#include "turbo/hash/fwd.h"
#include "turbo/hash/mix/common_mix.h"
#ifdef TURBO_HAVE_INTRINSIC_INT128
#include "turbo/base/int128.h"
#endif

namespace turbo::hash_internal {

    // Hash function for a byte array. A 64-bit seed and a set of five 64-bit
    // integers are hashed into the result.
    //
    // To allow all hashable types (including std::string_view and Span) to depend on
    // this algorithm, we keep the API low-level, with as few dependencies as
    // possible.
    uint64_t bytes_hash(const void *data, size_t len, uint64_t seed,
                          const uint64_t salt[5]);

}  // namespace turbo::hash_internal

namespace turbo {

    /**
     * @ingroup turbo_hash_engine
     * @brief bytes_hash_tag is a tag for bytes_hash.
     */
    struct bytes_hash_tag {

        static constexpr const char* name() {
            return "bytes_hash";
        }

        constexpr static bool available() {
            return true;
        }
    };

    template <>
    struct hasher_engine<bytes_hash_tag> {

        static uint64_t mix(uint64_t value);

        static uint64_t mix_with_seed(uint64_t seed, uint64_t value);

        static uint32_t hash32(const char *s, size_t len);

        static uint32_t hash32_with_seed(const char *s, size_t len, uint32_t seed);

        static size_t hash64(const char *s, size_t len);

        static size_t hash64_with_seed(const char *s, size_t len, uint64_t seed);

        // Seed()
        //
        // A non-deterministic seed.
        //
        // The current purpose of this seed is to generate non-deterministic results
        // and prevent having users depend on the particular hash values.
        // It is not meant as a security feature right now, but it leaves the door
        // open to upgrade it to a true per-process random seed. A true random seed
        // costs more and we don't need to pay for that right now.
        //
        // On platforms with ASLR, we take advantage of it to make a per-process
        // random value.
        // See https://en.wikipedia.org/wiki/Address_space_layout_randomization
        //
        // On other platforms this is still going to be non-deterministic but most
        // probably per-build and not per-process.
        TURBO_FORCE_INLINE static uint64_t Seed() {
#if (!defined(__clang__) || __clang_major__ > 11) && \
    !defined(__apple_build_version__)
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&kSeed));
#else
            // Workaround the absence of
                // https://github.com/llvm/llvm-project/commit/bc15bf66dcca76cc06fe71fca35b74dc4d521021.
                return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kSeed));
#endif
        }

    private:

        static const void *const kSeed;
    };


    inline uint64_t  hasher_engine<bytes_hash_tag>::mix(uint64_t value) {
        return turbo::hash_internal::common_mix_with_seed(Seed(), value);
    }
    inline uint64_t hasher_engine<bytes_hash_tag>::mix_with_seed(uint64_t seed, uint64_t value) {
        return turbo::hash_internal::common_mix_with_seed(seed, value);
    }
}  // namespace turbo
#endif  // TURBO_HASH_BYTES_LOW_LEVEL_HASH_H_
