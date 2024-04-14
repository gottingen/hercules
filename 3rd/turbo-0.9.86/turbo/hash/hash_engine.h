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

#ifndef TURBO_HASH_HASH_ENGINE_H_
#define TURBO_HASH_HASH_ENGINE_H_

#include "turbo/hash/fwd.h"
#include "turbo/hash/mix/murmur_mix.h"
#include "turbo/hash/mix/simple_mix.h"
#include "turbo/hash/bytes/city.h"
#include "turbo/hash/bytes/bytes_hash.h"
#include "turbo/hash/m3/m3.h"
#include "turbo/hash/xx/xx.h"
#include <map>
#include <string>

namespace turbo {

    /**
     * @ingroup turbo_hash_engine
     * @brief default_hash_engine is a default hash_engine for turbo::hash_engine.
     *        by default, it is bytes_hash. if you want to use other hash_engine,
     *        as the default hash_engine, you can define TURBO_DEFAUL_HASH_ENGINE to
     *        your hash_engine.
     *        for example:
     *        @code
     *        #define TURBO_DEFAUL_HASH_ENGINE city_hash
     *        #include <turbo/hash/hash.h>
     *        // now, the default hash_engine is city_hash.
     *        @endcode
     *        or
     *        @code
     *        #include <turbo/hash/hash.h>
     *        // now, the default hash_engine is bytes_hash.
     *        // specify the hash_engine when you use it.
     *        using my_hash_engine = turbo::hash_engine<bytes_hash_tag>;
     *        my_hash_engine hash_engine;
     *        size_t hash = hash_engine(key);
     *        @endcode
     *
     */
#ifdef TURBO_DEFAUT_HASH_ENGINE
using default_hash_engine = TURBO_DEFAUT_HASH_ENGINE;
#else
using default_hash_engine = bytes_hash_tag;
#endif

    constexpr const char* supported_hash_engines[] = {
        "bytes_hash",
        "m3_hash",
        "xx_hash",
    };
    enum class hash_engine_type {
        bytes_hash,
        m3_hash,
        xx_hash,
    };
    /**
     * for cmd line
     */
     extern std::map<std::string, hash_engine_type> engine_map;
}  // namespace turbo

#endif  // TURBO_HASH_HASH_ENGINE_H_
