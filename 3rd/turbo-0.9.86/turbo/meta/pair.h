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


#ifndef TURBO_META_PAIR_H_
#define TURBO_META_PAIR_H_

namespace turbo {


    // A custom pair implementation is used in the map because std::pair is not is_trivially_copyable,
    // which means it would  not be allowed to be used in std::memcpy. This struct is copyable, which is
    // also tested.
    template <typename T1, typename T2>
    struct pair {
        using first_type = T1;
        using second_type = T2;

        template <typename U1 = T1, typename U2 = T2,
                typename = typename std::enable_if<std::is_default_constructible<U1>::value &&
                                                   std::is_default_constructible<U2>::value>::type>
        constexpr pair() noexcept(noexcept(U1()) && noexcept(U2()))
                : first()
                , second() {}

        // pair constructors are explicit so we don't accidentally call this ctor when we don't have to.
        explicit constexpr pair(std::pair<T1, T2> const& o) noexcept(
        noexcept(T1(std::declval<T1 const&>())) && noexcept(T2(std::declval<T2 const&>())))
                : first(o.first)
                , second(o.second) {}

        // pair constructors are explicit so we don't accidentally call this ctor when we don't have to.
        explicit constexpr pair(std::pair<T1, T2>&& o) noexcept(noexcept(
                                                                        T1(std::move(std::declval<T1&&>()))) && noexcept(T2(std::move(std::declval<T2&&>()))))
                : first(std::move(o.first))
                , second(std::move(o.second)) {}

        constexpr pair(T1&& a, T2&& b) noexcept(noexcept(
                                                        T1(std::move(std::declval<T1&&>()))) && noexcept(T2(std::move(std::declval<T2&&>()))))
                : first(std::move(a))
                , second(std::move(b)) {}

        template <typename U1, typename U2>
        constexpr pair(U1&& a, U2&& b) noexcept(noexcept(T1(std::forward<U1>(
                std::declval<U1&&>()))) && noexcept(T2(std::forward<U2>(std::declval<U2&&>()))))
                : first(std::forward<U1>(a))
                , second(std::forward<U2>(b)) {}

        template <typename... U1, typename... U2>

        constexpr pair(std::piecewise_construct_t /*unused*/, std::tuple<U1...> a,
                       std::tuple<U2...>
                       b) noexcept(noexcept(pair(std::declval<std::tuple<U1...>&>(),
                                                 std::declval<std::tuple<U2...>&>(),
                                                 std::index_sequence_for<U1...>(),
                                                 std::index_sequence_for<U2...>())))
                : pair(a, b, std::index_sequence_for<U1...>(),
                       std::index_sequence_for<U2...>()) {
        }

        // constructor called from the std::piecewise_construct_t ctor
        template <typename... U1, size_t... I1, typename... U2, size_t... I2>
        pair(std::tuple<U1...>& a, std::tuple<U2...>& b, std::index_sequence<I1...> /*unused*/, std::index_sequence<I2...> /*unused*/) noexcept(
        noexcept(T1(std::forward<U1>(std::get<I1>(
                std::declval<std::tuple<
                        U1...>&>()))...)) && noexcept(T2(std::
                                                         forward<U2>(std::get<I2>(
                std::declval<std::tuple<U2...>&>()))...)))
                : first(std::forward<U1>(std::get<I1>(a))...)
                , second(std::forward<U2>(std::get<I2>(b))...) {
            // make visual studio compiler happy about warning about unused a & b.
            // Visual studio's pair implementation disables warning 4100.
            (void)a;
            (void)b;
        }

        void swap(pair<T1, T2>& o) noexcept(std::is_nothrow_swappable_v<T1>&&std::is_nothrow_swappable_v<T2>) {
            using std::swap;
            swap(first, o.first);
            swap(second, o.second);
        }

        T1 first;  // NOLINT(misc-non-private-member-variables-in-classes)
        T2 second; // NOLINT(misc-non-private-member-variables-in-classes)
    };

    template <typename A, typename B>
    inline void swap(pair<A, B>& a, pair<A, B>& b) noexcept(
    noexcept(std::declval<pair<A, B>&>().swap(std::declval<pair<A, B>&>()))) {
        a.swap(b);
    }

    template <typename A, typename B>
    inline constexpr bool operator==(pair<A, B> const& x, pair<A, B> const& y) {
        return (x.first == y.first) && (x.second == y.second);
    }
    template <typename A, typename B>
    inline constexpr bool operator!=(pair<A, B> const& x, pair<A, B> const& y) {
        return !(x == y);
    }
    template <typename A, typename B>
    inline constexpr bool operator<(pair<A, B> const& x, pair<A, B> const& y) noexcept(noexcept(
                                                                                               std::declval<A const&>() < std::declval<A const&>()) && noexcept(std::declval<B const&>() <
                                                                                                                                                                std::declval<B const&>())) {
        return x.first < y.first || (!(y.first < x.first) && x.second < y.second);
    }
    template <typename A, typename B>
    inline constexpr bool operator>(pair<A, B> const& x, pair<A, B> const& y) {
        return y < x;
    }
    template <typename A, typename B>
    inline constexpr bool operator<=(pair<A, B> const& x, pair<A, B> const& y) {
        return !(x > y);
    }
    template <typename A, typename B>
    inline constexpr bool operator>=(pair<A, B> const& x, pair<A, B> const& y) {
        return !(x < y);
    }

}  // namespace turbo

#endif  // TURBO_META_PAIR_H_
