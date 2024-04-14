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

#include "turbo/hash/bytes/bytes_hash.h"
#include "turbo/hash/bytes/city.h"
#include "turbo/platform/internal/unaligned_access.h"
#include "turbo/base/bits.h"

#ifdef TURBO_HAVE_INTRINSIC_INT128
namespace turbo::hash_internal {

    static uint64_t Mix(uint64_t v0, uint64_t v1) {
        turbo::uint128 p = v0;
        p *= v1;
        return turbo::uint128_low64(p) ^ turbo::uint128_high64(p);
    }

    uint64_t bytes_hash(const void *data, size_t len, uint64_t seed,
                          const uint64_t salt[5]) {
        const uint8_t *ptr = static_cast<const uint8_t *>(data);
        uint64_t starting_length = static_cast<uint64_t>(len);
        uint64_t current_state = seed ^ salt[0];

        if (len > 64) {
            // If we have more than 64 bytes, we're going to handle chunks of 64
            // bytes at a time. We're going to build up two separate hash states
            // which we will then hash together.
            uint64_t duplicated_state = current_state;

            do {
                uint64_t a = turbo::base_internal::UnalignedLoad64(ptr);
                uint64_t b = turbo::base_internal::UnalignedLoad64(ptr + 8);
                uint64_t c = turbo::base_internal::UnalignedLoad64(ptr + 16);
                uint64_t d = turbo::base_internal::UnalignedLoad64(ptr + 24);
                uint64_t e = turbo::base_internal::UnalignedLoad64(ptr + 32);
                uint64_t f = turbo::base_internal::UnalignedLoad64(ptr + 40);
                uint64_t g = turbo::base_internal::UnalignedLoad64(ptr + 48);
                uint64_t h = turbo::base_internal::UnalignedLoad64(ptr + 56);

                uint64_t cs0 = Mix(a ^ salt[1], b ^ current_state);
                uint64_t cs1 = Mix(c ^ salt[2], d ^ current_state);
                current_state = (cs0 ^ cs1);

                uint64_t ds0 = Mix(e ^ salt[3], f ^ duplicated_state);
                uint64_t ds1 = Mix(g ^ salt[4], h ^ duplicated_state);
                duplicated_state = (ds0 ^ ds1);

                ptr += 64;
                len -= 64;
            } while (len > 64);

            current_state = current_state ^ duplicated_state;
        }

        // We now have a data `ptr` with at most 64 bytes and the current state
        // of the hashing state machine stored in current_state.
        while (len > 16) {
            uint64_t a = turbo::base_internal::UnalignedLoad64(ptr);
            uint64_t b = turbo::base_internal::UnalignedLoad64(ptr + 8);

            current_state = Mix(a ^ salt[1], b ^ current_state);

            ptr += 16;
            len -= 16;
        }

        // We now have a data `ptr` with at most 16 bytes.
        uint64_t a = 0;
        uint64_t b = 0;
        if (len > 8) {
            // When we have at least 9 and at most 16 bytes, set A to the first 64
            // bits of the input and B to the last 64 bits of the input. Yes, they will
            // overlap in the middle if we are working with less than the full 16
            // bytes.
            a = turbo::base_internal::UnalignedLoad64(ptr);
            b = turbo::base_internal::UnalignedLoad64(ptr + len - 8);
        } else if (len > 3) {
            // If we have at least 4 and at most 8 bytes, set A to the first 32
            // bits and B to the last 32 bits.
            a = turbo::base_internal::UnalignedLoad32(ptr);
            b = turbo::base_internal::UnalignedLoad32(ptr + len - 4);
        } else if (len > 0) {
            // If we have at least 1 and at most 3 bytes, read all of the provided
            // bits into A, with some adjustments.
            a = static_cast<uint64_t>((ptr[0] << 16) | (ptr[len >> 1] << 8) |
                                      ptr[len - 1]);
            b = 0;
        } else {
            a = 0;
            b = 0;
        }

        uint64_t w = Mix(a ^ salt[1], b ^ current_state);
        uint64_t z = salt[1] ^ starting_length;
        return Mix(w, z);
    }
}  // namespace turbo::hash_internal
#endif // TURBO_HAVE_INTRINSIC_INT128

namespace turbo {
    namespace bytes_internal {

        // Reads 1 to 3 bytes from p. Zero pads to fill uint32_t.
        static uint32_t Read1To3(const unsigned char *p, size_t len) {
            unsigned char mem0 = p[0];
            unsigned char mem1 = p[len / 2];
            unsigned char mem2 = p[len - 1];
#if TURBO_IS_LITTLE_ENDIAN
            unsigned char significant2 = mem2;
            unsigned char significant1 = mem1;
            unsigned char significant0 = mem0;
#else
            unsigned char significant2 = mem0;
                unsigned char significant1 = mem1;
                unsigned char significant0 = mem2;
#endif
            return static_cast<uint32_t>(significant0 |                     //
                                         (significant1 << (len / 2 * 8)) |  //
                                         (significant2 << ((len - 1) * 8)));
        }

        constexpr uint64_t kDefaultHashSalt[5] = {
                uint64_t{0x243F6A8885A308D3}, uint64_t{0x13198A2E03707344},
                uint64_t{0xA4093822299F31D0}, uint64_t{0x082EFA98EC4E6C89},
                uint64_t{0x452821E638D01377},
        };

        // Reads 4 to 8 bytes from p. Zero pads to fill uint64_t.
        static uint64_t Read4To8(const unsigned char *p, size_t len) {
            uint32_t low_mem = turbo::base_internal::UnalignedLoad32(p);
            uint32_t high_mem = turbo::base_internal::UnalignedLoad32(p + len - 4);
#if TURBO_IS_LITTLE_ENDIAN
            uint32_t most_significant = high_mem;
            uint32_t least_significant = low_mem;
#else
            uint32_t most_significant = low_mem;
                uint32_t least_significant = high_mem;
#endif
            return (static_cast<uint64_t>(most_significant) << (len - 4) * 8) |
                   least_significant;
        }
        // Reads 9 to 16 bytes from p.
        // The least significant 8 bytes are in .first, the rest (zero padded) bytes
        // are in .second.
        static std::pair<uint64_t, uint64_t> Read9To16(const unsigned char *p,
                                                       size_t len) {
            uint64_t low_mem = turbo::base_internal::UnalignedLoad64(p);
            uint64_t high_mem = turbo::base_internal::UnalignedLoad64(p + len - 8);
#if TURBO_IS_LITTLE_ENDIAN
            uint64_t most_significant = high_mem;
            uint64_t least_significant = low_mem;
#else
            uint64_t most_significant = low_mem;
                uint64_t least_significant = high_mem;
#endif
            return {least_significant, most_significant};
        }


    } // namespace bytes_internal

    uint32_t hasher_engine<bytes_hash_tag>::hash32(const char *s, size_t len) {
        uint64_t v = 0;
        if (len > 8) {
            return hash_internal::CityHash32(s, len);
        } else if (len >= 4) {
            v = bytes_internal::Read4To8(reinterpret_cast<const unsigned char *>(s), len);
        } else if (len > 0) {
            v = bytes_internal::Read1To3(reinterpret_cast<const unsigned char *>(s), len);
        }
        return v;

    }

    uint32_t hasher_engine<bytes_hash_tag>::hash32_with_seed(const char *s, size_t len, uint32_t seed) {
        auto h = hash32(s, len);
        return h ? mix_with_seed(static_cast<uint64_t>(seed), h) : seed;
    }
    size_t hasher_engine<bytes_hash_tag>::hash64(const char *s, size_t len) {
        uint64_t v = 0;
        if (len > 16) {
#ifdef TURBO_HAVE_INTRINSIC_INT128
            return hash_internal::bytes_hash(s, len, 0, bytes_internal::kDefaultHashSalt);
#else
            return hash_internal::CityHash64(reinterpret_cast<const char*>(data), len);
#endif
        } else if (len > 8) {
            // This hash function was constructed by the ML-driven algorithm discovery
            // using reinforcement learning. We fed the agent lots of inputs from
            // microbenchmarks, SMHasher, low hamming distance from generated inputs and
            // picked up the one that was good on micro and macrobenchmarks.
            auto p = bytes_internal::Read9To16(reinterpret_cast<const unsigned char *>(s), len);
            uint64_t lo = p.first;
            uint64_t hi = p.second;
            return mix_with_seed(hi, lo);
        } else if (len >= 4) {
            v = bytes_internal::Read4To8(reinterpret_cast<const unsigned char *>(s), len);
        } else if (len > 0) {
            v = bytes_internal::Read1To3(reinterpret_cast<const unsigned char *>(s), len);
        }
        return v;

    }

    size_t hasher_engine<bytes_hash_tag>::hash64_with_seed(const char *s, size_t len, uint64_t seed) {
        // For large values we use LowLevelHash or CityHash depending on the platform,
        // for small ones we just use a multiplicative hash.
        uint64_t v;
        if (len > 16) {
#ifdef TURBO_HAVE_INTRINSIC_INT128
            return hash_internal::bytes_hash(s, len, seed, bytes_internal::kDefaultHashSalt);
#else
            return hash_internal::CityHash64WithSeed(reinterpret_cast<const char*>(data), len, seed);
#endif
        } else if (len > 8) {
            // This hash function was constructed by the ML-driven algorithm discovery
            // using reinforcement learning. We fed the agent lots of inputs from
            // microbenchmarks, SMHasher, low hamming distance from generated inputs and
            // picked up the one that was good on micro and macrobenchmarks.
            auto p = bytes_internal::Read9To16(reinterpret_cast<const unsigned char *>(s), len);
            uint64_t lo = p.first;
            uint64_t hi = p.second;
            // Rotation by 53 was found to be most often useful when discovering these
            // hashing algorithms with ML techniques.
            lo = turbo::rotr(lo, 53);
            seed += hash_internal::kMul;
            lo += seed;
            seed ^= hi;
            uint128 m = seed;
            m *= lo;
            return static_cast<uint64_t>(m ^ (m >> 64));
        } else if (len >= 4) {
            v = bytes_internal::Read4To8(reinterpret_cast<const unsigned char *>(s), len);
        } else if (len > 0) {
            v = bytes_internal::Read1To3(reinterpret_cast<const unsigned char *>(s), len);
        } else {
            // Empty ranges have no effect.
            return seed;
        }
        return mix_with_seed(seed, v);
    }

    TURBO_CONST_INIT const void *const hasher_engine<bytes_hash_tag>::kSeed = &kSeed;
}