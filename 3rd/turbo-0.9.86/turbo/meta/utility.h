// Copyright 2020 The Turbo Authors.
//
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
// This header file contains C++11 versions of standard <utility> header
// abstractions available within C++14 and C++17, and are designed to be drop-in
// replacement for code compliant with C++14 and C++17.
//
// The following abstractions are defined:
//
//   * integer_sequence<T, Ints...>  == std::integer_sequence<T, Ints...>
//   * index_sequence<Ints...>       == std::index_sequence<Ints...>
//   * make_integer_sequence<T, N>   == std::make_integer_sequence<T, N>
//   * make_index_sequence<N>        == std::make_index_sequence<N>
//   * index_sequence_for<Ts...>     == std::index_sequence_for<Ts...>
//   * apply<Functor, Tuple>         == std::apply<Functor, Tuple>
//   * exchange<T>                   == std::exchange<T>
//   * make_from_tuple<T>            == std::make_from_tuple<T>
//
// This header file also provides the tag types `in_place_t`, `in_place_type_t`,
// and `in_place_index_t`, as well as the constant `in_place`, and
// `constexpr` `std::move()` and `std::forward()` implementations in C++11.
//
// References:
//
//  https://en.cppreference.com/w/cpp/utility/integer_sequence
//  https://en.cppreference.com/w/cpp/utility/apply
//  http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3658.html

#ifndef TURBO_UTILITY_UTILITY_H_
#define TURBO_UTILITY_UTILITY_H_

#include <cstddef>
#include <cstdlib>
#include <tuple>
#include <utility>

#include "turbo/base/internal/inline_variable.h"
#include "turbo/meta/type_traits.h"
#include "turbo/platform/port.h"

namespace turbo {

    namespace utility_internal {
        // Helper method for expanding tuple into a called method.
        template<typename Functor, typename Tuple, std::size_t... Indexes>
        auto apply_helper(Functor &&functor, Tuple &&t, std::index_sequence<Indexes...>)
        -> decltype(std::invoke(
                std::forward<Functor>(functor),
                std::get<Indexes>(std::forward<Tuple>(t))...)) {
            return std::invoke(
                    std::forward<Functor>(functor),
                    std::get<Indexes>(std::forward<Tuple>(t))...);
        }

    }  // namespace utility_internal

    // apply
    //
    // Invokes a Callable using elements of a tuple as its arguments.
    // Each element of the tuple corresponds to an argument of the call (in order).
    // Both the Callable argument and the tuple argument are perfect-forwarded.
    // For member-function Callables, the first tuple element acts as the `this`
    // pointer. `turbo::apply` is designed to be a drop-in replacement for C++17's
    // `std::apply`. Unlike C++17's `std::apply`, this is not currently `constexpr`.
    //
    // Example:
    //
    //   class Foo {
    //    public:
    //     void Bar(int);
    //   };
    //   void user_function1(int, std::string);
    //   void user_function2(std::unique_ptr<Foo>);
    //   auto user_lambda = [](int, int) {};
    //
    //   int main()
    //   {
    //       std::tuple<int, std::string> tuple1(42, "bar");
    //       // Invokes the first user function on int, std::string.
    //       turbo::apply(&user_function1, tuple1);
    //
    //       std::tuple<std::unique_ptr<Foo>> tuple2(std::make_unique<Foo>());
    //       // Invokes the user function that takes ownership of the unique
    //       // pointer.
    //       turbo::apply(&user_function2, std::move(tuple2));
    //
    //       auto foo = std::make_unique<Foo>();
    //       std::tuple<Foo*, int> tuple3(foo.get(), 42);
    //       // Invokes the method Bar on foo with one argument, 42.
    //       turbo::apply(&Foo::Bar, tuple3);
    //
    //       std::tuple<int, int> tuple4(8, 9);
    //       // Invokes a lambda.
    //       turbo::apply(user_lambda, tuple4);
    //   }
    template<typename Functor, typename Tuple>
    auto apply(Functor &&functor, Tuple &&t)
    -> decltype(utility_internal::apply_helper(
            std::forward<Functor>(functor), std::forward<Tuple>(t),
            std::make_index_sequence<std::tuple_size<
                    typename std::remove_reference<Tuple>::type>::value>{})) {
        return utility_internal::apply_helper(
                std::forward<Functor>(functor), std::forward<Tuple>(t),
                std::make_index_sequence<std::tuple_size<
                        typename std::remove_reference<Tuple>::type>::value>{});
    }

    // exchange
    //
    // Replaces the value of `obj` with `new_value` and returns the old value of
    // `obj`.  `turbo::exchange` is designed to be a drop-in replacement for C++14's
    // `std::exchange`.
    //
    // Example:
    //
    //   Foo& operator=(Foo&& other) {
    //     ptr1_ = turbo::exchange(other.ptr1_, nullptr);
    //     int1_ = turbo::exchange(other.int1_, -1);
    //     return *this;
    //   }
    template<typename T, typename U = T>
    T exchange(T &obj, U &&new_value) {
        T old_value = std::move(obj);
        obj = std::forward<U>(new_value);
        return old_value;
    }

    namespace utility_internal {
        template<typename T, typename Tuple, size_t... I>
        T make_from_tuple_impl(Tuple &&tup, std::index_sequence<I...>) {
            return T(std::get<I>(std::forward<Tuple>(tup))...);
        }
    }  // namespace utility_internal

    // make_from_tuple
    //
    // Given the template parameter type `T` and a tuple of arguments
    // `std::tuple(arg0, arg1, ..., argN)` constructs an object of type `T` as if by
    // calling `T(arg0, arg1, ..., argN)`.
    //
    // Example:
    //
    //   std::tuple<const char*, size_t> args("hello world", 5);
    //   auto s = turbo::make_from_tuple<std::string>(args);
    //   assert(s == "hello");
    //
    template<typename T, typename Tuple>
    constexpr T make_from_tuple(Tuple &&tup) {
        return utility_internal::make_from_tuple_impl<T>(
                std::forward<Tuple>(tup),
                std::make_index_sequence<
                        std::tuple_size<std::decay_t<Tuple>>::value>{});
    }

    template<typename Int>
    constexpr auto to_unsigned(Int value) ->
    typename std::make_unsigned<Int>::type {
        TURBO_ASSERT((std::is_unsigned<Int>::value || value >= 0)&&"negative value");
        return static_cast<typename std::make_unsigned<Int>::type>(value);
    }
}  // namespace turbo

#endif  // TURBO_UTILITY_UTILITY_H_
