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


#ifndef TURBO_HASH_HASH_H_
#define TURBO_HASH_HASH_H_

#include "turbo/hash/hash_engine.h"

namespace turbo {

    /**
     * @ingroup turbo_hash_mixer
     * @brief default_mixer is a default mixer for turbo::Mixer.
     *        by default, it is simple_mix. if you want to use other mixer,
     *        as the default mixer, you can define TURBO_DEFAUL_MIXER to
     *        your mixer.
     *        for example:
     *        @code
     *        #define TURBO_DEFAUL_MIXER murmur_mix
     *        #include <turbo/hash/hash.h>
     *        // now, the default mixer is murmur_mix.
     *        @endcode
     *        or
     *        @code
     *        #include <turbo/hash/hash.h>
     *        // now, the default mixer is simple_mix.
     *        // specify the mixer when you use it.
     *        using my_mixer = turbo::Mixer<4, size_t, murmur_mix>;
     *        my_mixer mixer;
     *        size_t hash = mixer(key);
     *        @endcode
     *
     */
#ifdef TURBO_DEFAUL_MIXER
    using default_mixer = TURBO_DEFAUL_MIXER;
#else
    using default_mixer = simple_mix;
#endif

    template<int N, typename E>
    struct hash_mixer_traits {
        using type = typename std::conditional<N == 4,
                typename E::mix4, typename E::mix8>::type;
    };

    /**
     * @ingroup turbo_hash_mixer
     * @brief Mixer is a hash mixer for integer.
     *        it is a wrapper of hash_mixer_traits. it is used to
     *        specify the size of integer and return type.
     *        for example:
     *        @code
     *        #include <turbo/hash/hash.h>
     *        size_t key = 0x12345678;
     *        using my_mixer = turbo::Mixer<4, size_t, murmur_mix>;
     *        my_mixer mixer;
     *        size_t hash = mixer(key);
     *        @endcode
     * @tparam N size of integer
     * @tparam R return type
     * @tparam Engine hash mixer engine
     */
    template<int N, typename R = size_t, typename Engine = default_mixer>
    struct Mixer {
        static_assert(sizeof(R) >= N, "R must be larger than N");
        static_assert(N == 4 || N == 8, "N must be 4 or 8");
        constexpr R operator()(size_t key) const {
            using engine_type = typename hash_mixer_traits<N, Engine>::type;
            return static_cast<R>(engine_type::mix(key));
        }
    };

    /**
     * @ingroup turbo_hash_mixer
     * @brief hash_mixer4 is a hash mixer for 4 bytes integer.
     *        it is a wrapper of Mixer<4, R, Engine>.
     *        for example:
     *        @code
     *        #include <turbo/hash/hash.h>
     *        size_t key = 0x12345678;
     *        auto hash = turbo::hash_mixer4<uint32_t>(key);
     *        @endcode
     * @tparam R return type
     * @tparam Engine hash mixer engine
     */
    template<typename R, typename Engine = default_mixer>
    constexpr typename std::enable_if_t<!std::is_same_v<R, void> && std::is_integral_v<R> &&!is_mix_engine<R>::value ,R>
            hash_mixer4(size_t key) {
        return Mixer<4, R, Engine>()(key);
    }

    /**
     * @ingroup turbo_hash_mixer
     * @brief hash_mixer4 is a hash mixer for 4 bytes integer.
     *        overload for default return type.
     * @see   hash_mixer4<R, Engine>
     * @tparam Engine hash mixer engine
     */
    template<typename Engine = default_mixer, std::enable_if_t<is_mix_engine<Engine>::value , int> = 0>
    constexpr size_t hash_mixer4(size_t key) {
        return Mixer<4, size_t, Engine>()(key);
    }

    /**
     * @ingroup turbo_hash_mixer
     * @brief hash_mixer8 is a hash mixer for 8 bytes integer.
     *        it is a wrapper of Mixer<8, R, Engine>.
     *        for example:
     *        @code
     *        #include <turbo/hash/hash.h>
     *        size_t key = 0x12345678;
     *        auto hash = turbo::hash_mixer8<uint32_t>(key);
     *        @endcode
     * @tparam R return type
     * @tparam Engine hash mixer engine
     */
    template<typename R, typename Engine = default_mixer>
    constexpr typename std::enable_if_t<!std::is_same_v<R, void> && std::is_integral_v<R> &&!is_mix_engine<R>::value ,R>
    hash_mixer8(size_t key) {
        return Mixer<8, R, Engine>()(key);
    }

    /**
     * @ingroup turbo_hash_mixer
     * @brief hash_mixer8 is a hash mixer for 8 bytes integer.
     *        overload for default return type.
     * @see   hash_mixer8<R, Engine>
     * @tparam Engine hash mixer engine
     */
    template<typename Engine = default_mixer, std::enable_if_t<is_mix_engine<Engine>::value , int> = 0>
    constexpr size_t hash_mixer8(size_t key) {
        return Mixer<8, size_t, Engine>()(key);
    }


}  // namespace turbo

#endif  // TURBO_HASH_HASH_H_
