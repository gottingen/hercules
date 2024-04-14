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

#include "turbo/hash/m3/m3.h"
#include "turbo/hash/m3/murmurhash3.h"

namespace turbo {
    uint32_t hasher_engine<m3_hash_tag>::hash32(const char *s, size_t len) {
        uint32_t hash;
        hash_internal::MurmurHash3_x86_32_Context ctx;
        hash_internal::MurmurHash3_x86_32_Init(&ctx, 0);
        hash_internal::MurmurHash3_x86_32_Update(&ctx, s, len);
        hash_internal::MurmurHash3_x86_32_Final(&hash, &ctx);
        return hash;
    }

    uint32_t hasher_engine<m3_hash_tag>::hash32_with_seed(const char *s, size_t len, uint32_t seed) {
        uint32_t hash;
        hash_internal::MurmurHash3_x86_32_Context ctx;
        hash_internal::MurmurHash3_x86_32_Init(&ctx, seed);
        hash_internal::MurmurHash3_x86_32_Update(&ctx, s, len);
        hash_internal::MurmurHash3_x86_32_Final(&hash, &ctx);
        return hash;
    }

    size_t hasher_engine<m3_hash_tag>::hash64(const char *s, size_t len) {
        uint64_t hash[2];
        hash_internal::MurmurHash3_x64_128_Context ctx;
        hash_internal::MurmurHash3_x64_128_Init(&ctx, 0);
        hash_internal::MurmurHash3_x64_128_Update(&ctx, s, len);
        hash_internal::MurmurHash3_x64_128_Final(&hash, &ctx);
        return hash[0] + hash[1];
    }

    size_t hasher_engine<m3_hash_tag>::hash64_with_seed(const char *s, size_t len, uint64_t seed) {
        uint64_t hash[2];
        hash_internal::MurmurHash3_x64_128_Context ctx;
        hash_internal::MurmurHash3_x64_128_Init(&ctx, seed);
        hash_internal::MurmurHash3_x64_128_Update(&ctx, s, len);
        hash_internal::MurmurHash3_x64_128_Final(&hash, &ctx);
        return hash[0] + hash[1];
    }
}