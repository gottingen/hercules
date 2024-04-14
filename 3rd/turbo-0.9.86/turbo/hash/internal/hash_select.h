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
// Created by jeff on 23-12-12.
//

#ifndef TURBO_HASH_INTERNAL_HASH_SELECT_H_
#define TURBO_HASH_INTERNAL_HASH_SELECT_H_

#include "turbo/hash/internal/hash_state_base.h"

namespace turbo::hash_internal {
    // HashSelect
    //
    // Type trait to select the appropriate hash implementation to use.
    // HashSelect::type<T> will give the proper hash implementation, to be invoked
    // as:
    //   HashSelect::type<T>::Invoke(state, value)
    // Also, HashSelect::type<T>::value is a boolean equal to `true` if there is a
    // valid `Invoke` function. Types that are not hashable will have a ::value of
    // `false`.
    struct HashSelect {
    private:
        struct State : HashStateBase<State> {
            static State combine_contiguous(State hash_state, const unsigned char *,
                                            size_t);

            using State::HashStateBase::combine_contiguous;
        };

        struct UniquelyRepresentedProbe {
            template<typename H, typename T>
            static auto Invoke(H state, const T &value)
            -> std::enable_if_t<is_uniquely_represented<T>::value, H> {
                return hash_internal::hash_bytes(std::move(state), value);
            }
        };

        struct HashValueProbe {
            template<typename H, typename T>
            static auto Invoke(H state, const T &value) -> std::enable_if_t<
            std::is_same<H,
                    decltype(hash_value(std::move(state), value))>::value,
            H> {
                return hash_value(std::move(state), value);
            }
        };

        struct LegacyHashProbe {
#if TURBO_HASH_INTERNAL_SUPPORT_LEGACY_HASH_
            template <typename H, typename T>
                static auto Invoke(H state, const T& value) -> std::enable_if_t<
                    std::is_convertible<
                        decltype(TURBO_INTERNAL_LEGACY_HASH_NAMESPACE::hash<T>()(value)),
                        size_t>::value,
                    H> {
                  return hash_internal::hash_bytes(
                      std::move(state),
                      TURBO_INTERNAL_LEGACY_HASH_NAMESPACE::hash<T>{}(value));
                }
#endif  // TURBO_HASH_INTERNAL_SUPPORT_LEGACY_HASH_
        };

        struct StdHashProbe {
            template<typename H, typename T>
            static auto Invoke(H state, const T &value)
            -> std::enable_if_t<type_traits_internal::IsHashable<T>::value, H> {
                return hash_internal::hash_bytes(std::move(state), std::hash<T>{}(value));
            }
        };

        template<typename Hash, typename T>
        struct Probe : Hash {
        private:
            template<typename H, typename = decltype(H::Invoke(
                    std::declval<State>(), std::declval<const T &>()))>
            static std::true_type Test(int);

            template<typename U>
            static std::false_type Test(char);

        public:
            static constexpr bool value = decltype(Test<Hash>(0))::value;
        };

    public:
        // Probe each implementation in order.
        // disjunction provides short circuiting wrt instantiation.
        template<typename T>
        using Apply = std::disjunction<         //
                Probe<UniquelyRepresentedProbe, T>,  //
                Probe<HashValueProbe, T>,            //
                Probe<LegacyHashProbe, T>,           //
                Probe<StdHashProbe, T>,              //
                std::false_type>;
    };

    template<typename T>
    struct is_hashable
            : std::integral_constant<bool, HashSelect::template Apply<T>::value> {
    };
}  // namespace turbo::hash_internal

#endif  // TURBO_HASH_INTERNAL_HASH_SELECT_H_
