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
//
// Created by jeff on 23-12-11.
//

#ifndef TURBO_HASH_FWD_H_
#define TURBO_HASH_FWD_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace turbo {

    template<int n>
    struct fold_if_needed {
        inline size_t operator()(uint64_t) const;
    };

    template<>
    struct fold_if_needed<4> {
        inline size_t operator()(uint64_t a) const {
            return static_cast<size_t>(a ^ (a >> 32));
        }
    };

    template<>
    struct fold_if_needed<8> {
        inline size_t operator()(uint64_t a) const {
            return static_cast<size_t>(a);
        }
    };

    template<typename T>
    struct has_hash_value {
    private:
        typedef std::true_type yes;
        typedef std::false_type no;

        template<typename U>
        static auto test(int) -> decltype(hash_value(std::declval<const U &>()) == 1, yes());

        template<typename>
        static no test(...);

    public:
        static constexpr bool value = std::is_same<decltype(test<T>(0)), yes>::value;
    };

    /// forward declarations

    template<int n>
    struct MixEngine {
        static constexpr size_t mix(size_t key);
    };

    template<typename E>
    struct is_mix_engine : public std::false_type {};

    /**
     * @brief interface for hash engine
     * @tparam Tag
     */
    struct hash_engine_tag {

        constexpr static const char* name();

        constexpr static bool available();
    };

    /**
     * @ingroup turbo_hash_engine
     * @brief hasher_engine is a hash engine, which is a wrapper of hash function.
     *        It provides a uniform interface for hash functions. It is specialized
     *        by the engine tags.
     *        The engine tags are defined in turbo/hash/xx/xx.h, turbo/hash/m3/m3.h,
     *        turbo/hash/city/city.h, turbo/hash/bytes/bytes_hash.h.
     *        user can define their own engine tags and specialize hasher_engine.
     *        Example:
     *        @code
     *        // I want to use xx_hash as my hash function
     *        auto hash_value = turbo::hasher_engine<turbo::xx_hash_tag>::hash64("hello", 5);
     *        @endcode
     *
     *        another example:
     *        @code
     *        // I want to use xx_hash as my hash function
     *        auto hash_value = turbo::Hash<uint64_t, turbo::xx_hash_tag>("hello", 5);
     *        @endcode
     *        the above two examples are equivalent. you can use the one you like.
     *        the second example is more flexible, you can use it to hash any type of
     *        data, as long as you have defined a hash_value function for it.
     *        Example:
     *        @code
     *        // I want to use xx_hash as my hash function
     *        struct MyStruct {
     *          int a;
     *          int b;
     *
     *          template <typename H>
     *          friend H hash_value(H &&h, const MyStruct& s) {
     *              H::combine(std::move(h), s.a, s.b);
     *          }
     *        };
     *        // now you can use turbo::Hash to hash MyStruct
     *        auto hash_value = turbo::Hash<MyStruct, turbo::xx_hash_tag>(MyStruct{1, 2});
     *        @endcode
     *        when you specialize hasher_engine, turbo actually call hasher_engine<tag> for
     *        you. so you can define your own hasher_engine like this:
     *        @code
     *        namespace turbo {
     *
     *            struct my_hash_tag {
     *                static constexpr const char* name() {
     *                    return "my_hash";
     *                 }
     *
     *                 constexpr static bool available() {
     *                    return true;
     *                  }
     *            };
     *
     *            template <>
     *            struct hasher_engine<my_hash_tag> {
     *                static uint32_t hash32(const char *s, size_t len) {
     *                // your hash function
     *                // just return a number to test
     *                return 1;
     *                }
     *
     *                static size_t hash64(const char *s, size_t len) {
     *                    return 2;
     *                    // your hash function
     *                    // just return a number to test
     *                 }
     *
     *                 static size_t hash64_with_seed(const char *s, size_t len, uint64_t seed) {
     *                     return 3;
     *                     // your hash function
     *                     // just return a number to test
     *                  }
     *             };
     *
     *             // now you can use turbo::Hash to hash MyStruct
     *
     *             auto hash_value = turbo::Hash<MyStruct, turbo::my_hash_tag>(MyStruct{1, 2});
     *             @endcode
     *
     * @tparam Tag
     */
    template <typename Tag>
    struct hasher_engine;

}  // namespace turbo

#endif  // TURBO_HASH_FWD_H_
