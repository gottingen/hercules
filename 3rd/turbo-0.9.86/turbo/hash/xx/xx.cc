// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
#include "turbo/platform/port.h"
#include "turbo/hash/xx/xx.h"
#include "turbo/format/print.h"
#if TURBO_WITH_AVX2
namespace turbo {
    uint32_t hasher_engine<xx_hash_tag>::hash32(const char *s, size_t len) {
        auto* ctx = XXH3_createState();
        if(!ctx) {
            return 0;
        }
        XXH3_64bits_reset(ctx);
        XXH3_64bits_update(ctx, s, len);
        uint32_t hash = XXH3_64bits_digest(ctx);
        XXH3_freeState(ctx);
        return hash;
    }

    size_t hasher_engine<xx_hash_tag>::hash32_with_seed(const char *s, size_t len, uint32_t seed) {
        auto* ctx = XXH3_createState();
        if(!ctx) {
            return 0;
        }
        XXH3_64bits_reset_withSeed(ctx, seed);
        XXH3_64bits_update(ctx, s, len);
        uint32_t hash = XXH3_64bits_digest(ctx);
        XXH3_freeState(ctx);
        return hash;
    }
    size_t hasher_engine<xx_hash_tag>::hash64(const char *s, size_t len) {
        auto* ctx = XXH3_createState();
        if(!ctx) {
            return 0;
        }
        XXH3_128bits_reset(ctx);
        XXH3_128bits_update(ctx, s, len);
        XXH128_hash_t hash = XXH3_128bits_digest(ctx);
        XXH3_freeState(ctx);
        return hash.high64 + hash.low64;
    }

    size_t hasher_engine<xx_hash_tag>::hash64_with_seed(const char *s, size_t len, uint64_t seed) {
        auto* ctx = XXH3_createState();
        if(!ctx) {
            return 0;
        }
        XXH3_128bits_reset_withSeed(ctx, seed);
        XXH3_128bits_update(ctx, s, len);
        XXH128_hash_t hash = XXH3_128bits_digest(ctx);
        XXH3_freeState(ctx);
        return hash.high64 + hash.low64;
    }
}
#else
namespace turbo {
    uint32_t hasher_engine<xx_hash_tag>::hash32(const char *s, size_t len) {
        auto* ctx = XXH32_createState();
        if(!ctx) {
            return 0;
        }
        XXH32_reset(ctx, 0);
        XXH32_update(ctx, s, len);
        uint32_t hash = XXH32_digest(ctx);
        XXH32_freeState(ctx);
        return hash;
    }

    size_t hasher_engine<xx_hash_tag>::hash64(const char *s, size_t len) {
        auto* ctx = XXH64_createState();
        if(!ctx) {
            return 0;
        }
        XXH64_reset(ctx, 0);
        XXH64_update(ctx, s, len);
        uint64_t hash = XXH64_digest(ctx);
        XXH64_freeState(ctx);
        return hash;
    }

    size_t hasher_engine<xx_hash_tag>::hash64_with_seed(const char *s, size_t len, uint64_t seed) {
        auto* ctx = XXH64_createState();
        if(!ctx) {
            return 0;
        }
        XXH64_reset(ctx, seed);
        XXH64_update(ctx, s, len);
        uint64_t hash = XXH64_digest(ctx);
        XXH64_freeState(ctx);
        return hash;
    }
}
#endif