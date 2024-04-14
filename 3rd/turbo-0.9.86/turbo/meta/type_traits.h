//
// Copyright 2023 The Turbo Authors.
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

#ifndef TURBO_META_TYPE_TRAITS_H_
#define TURBO_META_TYPE_TRAITS_H_

#include <cstddef>
#include <functional>
#include <type_traits>
#include <string>
#include <complex>
#include <chrono>
#include "turbo/platform/port.h"

// MSVC constructibility traits do not detect destructor properties and so our
// implementations should not use them as a source-of-truth.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__GNUC__)
#define TURBO_META_INTERNAL_STD_CONSTRUCTION_TRAITS_DONT_CHECK_DESTRUCTION 1
#endif

// Defines the default alignment. `__STDCPP_DEFAULT_NEW_ALIGNMENT__` is a C++17
// feature.
#if defined(__STDCPP_DEFAULT_NEW_ALIGNMENT__)
#define TURBO_INTERNAL_DEFAULT_NEW_ALIGNMENT __STDCPP_DEFAULT_NEW_ALIGNMENT__
#else  // defined(__STDCPP_DEFAULT_NEW_ALIGNMENT__)
#define TURBO_INTERNAL_DEFAULT_NEW_ALIGNMENT alignof(std::max_align_t)
#endif  // defined(__STDCPP_DEFAULT_NEW_ALIGNMENT__)

namespace turbo {

#if defined(__cpp_lib_remove_cvref) && __cpp_lib_remove_cvref >= 201711L
    template <typename T>
    using remove_cvref = std::remove_cvref<T>;

    template <typename T>
    using remove_cvref_t = typename std::remove_cvref<T>::type;
#else
    // remove_cvref()
    //
    // C++11 compatible implementation of std::remove_cvref which was added in
    // C++20.
    template<typename T>
    struct remove_cvref {
        using type =
                typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    };

    template<typename T>
    using remove_cvref_t = typename remove_cvref<T>::type;
#endif


    namespace type_traits_internal {
// In MSVC we can't probe std::hash or stdext::hash because it triggers a
// static_assert instead of failing substitution. Libc++ prior to 4.0
// also used a static_assert.
//
#if defined(_MSC_VER) || (defined(_LIBCPP_VERSION) && \
                          _LIBCPP_VERSION < 4000 && _LIBCPP_STD_VER > 11)
#define TURBO_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_ 0
#else
#define TURBO_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_ 1
#endif

#if !TURBO_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_
        template <typename Key, typename = size_t>
        struct IsHashable : std::true_type {};
#else   // TURBO_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_
        template<typename Key, typename = void>
        struct IsHashable : std::false_type {
        };

        template<typename Key>
        struct IsHashable<
                Key,
                std::enable_if_t<std::is_convertible<
                        decltype(std::declval<std::hash<Key> &>()(std::declval<Key const &>())),
                        std::size_t>::value>> : std::true_type {
        };
#endif  // !TURBO_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_
    }  // namespace type_traits_internal


// turbo::is_trivially_relocatable<T>
// Detects whether a type is "trivially relocatable" -- meaning it can be
// relocated without invoking the constructor/destructor, using a form of move
// elision.
//
// Example:
//
// if constexpr (turbo::is_trivially_relocatable<T>::value) {
//   memcpy(new_location, old_location, sizeof(T));
// } else {
//   new(new_location) T(std::move(*old_location));
//   old_location->~T();
// }
//
// Upstream documentation:
//
// https://clang.llvm.org/docs/LanguageExtensions.html#:~:text=__is_trivially_relocatable
//
#if TURBO_HAVE_BUILTIN(__is_trivially_relocatable)
    template <class T>
    struct is_trivially_relocatable
        : std::integral_constant<bool, __is_trivially_relocatable(T)> {};
#else
    template<class T>
    struct is_trivially_relocatable : std::integral_constant<bool, false> {
    };
#endif

    // turbo::is_constant_evaluated()
    //
    // Detects whether the function call occurs within a constant-evaluated context.
    // Returns true if the evaluation of the call occurs within the evaluation of an
    // expression or conversion that is manifestly constant-evaluated; otherwise
    // returns false.
    //
    // This function is implemented in terms of `std::is_constant_evaluated` for
    // c++20 and up. For older c++ versions, the function is implemented in terms
    // of `__builtin_is_constant_evaluated` if available, otherwise the function
    // will fail to compile.
    //
    // Applications can inspect `TURBO_HAVE_CONSTANT_EVALUATED` at compile time
    // to check if this function is supported.
    //
    // Example:
    //
    // constexpr MyClass::MyClass(int param) {
    // #ifdef TURBO_HAVE_CONSTANT_EVALUATED
    //   if (!turbo::is_constant_evaluated()) {
    //     TURBO_LOG(INFO) << "MyClass(" << param << ")";
    //   }
    // #endif  // TURBO_HAVE_CONSTANT_EVALUATED
    // }
    //
    // Upstream documentation:
    //
    // http://en.cppreference.com/w/cpp/types/is_constant_evaluated
    // http://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#:~:text=__builtin_is_constant_evaluated
    //
    constexpr TURBO_FORCE_INLINE auto is_constant_evaluated(bool default_value = false) noexcept -> bool {
        // Workaround for incompatibility between libstdc++ consteval-based
        // std::is_constant_evaluated() implementation and clang-14.
        // https://github.com/fmtlib/fmt/issues/3247
#if TURBO_CPLUSPLUS >= TURBO_CPLUSPLUS_20 && defined(_GLIBCXX_RELEASE) && \
    _GLIBCXX_RELEASE >= 12 && \
    (TURBO_CLANG_VERSION >= 1400 && TURBO_CLANG_VERSION < 1500)
        TURBO_UNUSED(default_value);
                return __builtin_is_constant_evaluated();
#elif defined(__cpp_lib_is_constant_evaluated)
        TURBO_UNUSED(default_value);
                return std::is_constant_evaluated();
#else
        return default_value;
#endif
    }


#ifdef TURBO_COMPILER_HAVE_RTTI
#define TURBO_TYPE_INFO_OF(...) (&typeid(__VA_ARGS__))
#else
#define TURBO_TYPE_INFO_OF(...) \
  ((sizeof(__VA_ARGS__)), static_cast<std::type_info const*>(nullptr))
#endif

    template<typename T>
    struct add_cr_non_integral {
        typedef typename std::conditional<std::is_integral<T>::value, T,
                typename std::add_lvalue_reference_t<typename std::add_const_t<T>>>::type type;
    };

    //  type_info_of
    //
    //  Returns &typeid(T) if RTTI is available, nullptr otherwise.
    //
    //  This overload works on the static type of the template parameter.
    template<typename T>
    TURBO_FORCE_INLINE static std::type_info const *type_info_of() {
        return TURBO_TYPE_INFO_OF(T);
    }

    //  type_info_of
    //
    //  Returns &typeid(t) if RTTI is available, nullptr otherwise.
    //
    //  This overload works on the dynamic type of the non-template parameter.
    template<typename T>
    TURBO_FORCE_INLINE static std::type_info const *type_info_of(TURBO_MAYBE_UNUSED T const &t) {
        return TURBO_TYPE_INFO_OF(t);
    }

    template<typename T>
    struct is_string_type : public std::false_type {
    };
    template<>
    struct is_string_type<std::string> : public std::true_type {
    };

    /************************************
    * arithmetic type promotion traits *
    ************************************/

    /**
     * Traits class for the result type of mixed arithmetic expressions.
     * For example, <tt>promote_type<unsigned char, unsigned char>::type</tt> tells
     * the user that <tt>unsigned char + unsigned char => int</tt>.
     */
    template<class... T>
    struct promote_type;

    template<>
    struct promote_type<> {
        using type = void;
    };

    template<class T>
    struct promote_type<T> {
        using type = typename promote_type<T, T>::type;
    };

    template<class C, class D1, class D2>
    struct promote_type<std::chrono::time_point<C, D1>, std::chrono::time_point<C, D2>> {
        using type = std::chrono::time_point<C, typename promote_type<D1, D2>::type>;
    };

    template<class T0, class T1>
    struct promote_type<T0, T1> {
        using type = decltype(std::declval<std::decay_t<T0>>() + std::declval<std::decay_t<T1>>());
    };

    template<class T0, class... REST>
    struct promote_type<T0, REST...> {
        using type = decltype(std::declval<std::decay_t<T0>>() + std::declval<typename promote_type<REST...>::type>());
    };

    template<>
    struct promote_type<bool> {
        using type = bool;
    };

    template<class T>
    struct promote_type<bool, T> {
        using type = T;
    };

    template<class T>
    struct promote_type<bool, std::complex<T>> {
        using type = std::complex<T>;
    };

    template<class T1, class T2>
    struct promote_type<T1, std::complex<T2>> {
        using type = std::complex<typename promote_type<T1, T2>::type>;
    };

    template<class T1, class T2>
    struct promote_type<std::complex<T1>, T2>
            : promote_type<T2, std::complex<T1>> {
    };

    template<class T>
    struct promote_type<std::complex<T>, std::complex<T>> {
        using type = std::complex<T>;
    };

    template<class T1, class T2>
    struct promote_type<std::complex<T1>, std::complex<T2>> {
        using type = std::complex<typename promote_type<T1, T2>::type>;
    };

    template<class... REST>
    struct promote_type<bool, REST...> {
        using type = typename promote_type<bool, typename promote_type<REST...>::type>::type;
    };

    /**
     * Abbreviation of 'typename promote_type<T>::type'.
     */
    template<class... T>
    using promote_type_t = typename promote_type<T...>::type;

    /**
     * Traits class to find the biggest type of the same kind.
     *
     * For example, <tt>big_promote_type<unsigned char>::type</tt> is <tt>unsigned long long</tt>.
     * The default implementation only supports built-in types and <tt>std::complex</tt>. All
     * other types remain unchanged unless <tt>big_promote_type</tt> gets specialized for them.
     */
    template<class T>
    struct big_promote_type {
    private:

        using V = std::decay_t<T>;
        static constexpr bool is_arithmetic = std::is_arithmetic<V>::value;
        static constexpr bool is_signed = std::is_signed<V>::value;
        static constexpr bool is_integral = std::is_integral<V>::value;
        static constexpr bool is_long_double = std::is_same<V, long double>::value;

    public:

        using type = std::conditional_t<is_arithmetic,
                std::conditional_t<is_integral,
                        std::conditional_t<is_signed, long long, unsigned long long>,
                        std::conditional_t<is_long_double, long double, double>
                >,
                V
        >;
    };

    template<class T>
    struct big_promote_type<std::complex<T>> {
        using type = std::complex<typename big_promote_type<T>::type>;
    };

    /**
     * Abbreviation of 'typename big_promote_type<T>::type'.
     */
    template<class T>
    using big_promote_type_t = typename big_promote_type<T>::type;

    namespace traits_detail {
        using std::sqrt;

        template<class T>
        using real_promote_type_t = decltype(sqrt(std::declval<std::decay_t<T>>()));
    }

    /**
     * Result type of algebraic expressions.
     *
     * For example, <tt>real_promote_type<int>::type</tt> tells the
     * user that <tt>sqrt(int) => double</tt>.
     */
    template<class T>
    struct real_promote_type {
        using type = traits_detail::real_promote_type_t<T>;
    };

    /**
     * Abbreviation of 'typename real_promote_type<T>::type'.
     */
    template<class T>
    using real_promote_type_t = typename real_promote_type<T>::type;

    /**
     * Traits class to replace 'bool' with 'uint8_t' and keep everything else.
     *
     * This is useful for scientific computing, where a boolean mask array is
     * usually implemented as an array of bytes.
     */
    template<class T>
    struct bool_promote_type {
        using type = typename std::conditional<std::is_same<T, bool>::value, uint8_t, T>::type;
    };

    /**
     * Abbreviation for typename bool_promote_type<T>::type
     */
    template<class T>
    using bool_promote_type_t = typename bool_promote_type<T>::type;

    /************
     * apply_cv *
     ************/

    namespace detail {
        template<class T, class U, bool = std::is_const<std::remove_reference_t<T>>::value,
                bool = std::is_volatile<std::remove_reference_t<T>>::value>
        struct apply_cv_impl {
            using type = U;
        };

        template<class T, class U>
        struct apply_cv_impl<T, U, true, false> {
            using type = const U;
        };

        template<class T, class U>
        struct apply_cv_impl<T, U, false, true> {
            using type = volatile U;
        };

        template<class T, class U>
        struct apply_cv_impl<T, U, true, true> {
            using type = const volatile U;
        };

        template<class T, class U>
        struct apply_cv_impl<T &, U, false, false> {
            using type = U &;
        };

        template<class T, class U>
        struct apply_cv_impl<T &, U, true, false> {
            using type = const U &;
        };

        template<class T, class U>
        struct apply_cv_impl<T &, U, false, true> {
            using type = volatile U &;
        };

        template<class T, class U>
        struct apply_cv_impl<T &, U, true, true> {
            using type = const volatile U &;
        };
    }

    template<class T, class U>
    struct apply_cv {
        using type = typename detail::apply_cv_impl<T, U>::type;
    };

    template<class T, class U>
    using apply_cv_t = typename apply_cv<T, U>::type;



    /************
     * concepts *
     ************/

#define TURBO_ENABLE_IF(...) std::enable_if_t<(__VA_ARGS__), int> = 0
#define TURBO_FORWARD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)

#if !defined(__GNUC__) || (defined(__GNUC__) && (__GNUC__ >= 5))

    template<class... C>
    constexpr bool turbo_require = std::conjunction<C...>::value;

    template<class... C>
    constexpr bool either = std::disjunction<C...>::value;

    template<class... C>
    constexpr bool disallow = std::negation<std::conjunction<C...>>::value;

    template<class... C>
    constexpr bool disallow_one = std::negation<std::disjunction<C...>>::value;

    template<class... C>
    using check_requires = std::enable_if_t<turbo_require<C...>, int>;

    template<class... C>
    using check_either = std::enable_if_t<either<C...>, int>;

    template<class... C>
    using check_disallow = std::enable_if_t<disallow<C...>, int>;

    template<class... C>
    using check_disallow_one = std::enable_if_t<disallow_one<C...>, int>;

#else

    template <class... C>
    using check_requires = std::enable_if_t<conjunction<C...>::value, int>;

    template <class... C>
    using check_either = std::enable_if_t<std::disjunction<C...>::value, int>;

    template <class... C>
    using check_disallow = std::enable_if_t<std::negation<std::conjunction<C...>>::value, int>;

    template <class... C>
    using check_disallow_one = std::enable_if_t<std::negation<std::disjunction<C...>>::value, int>;

#endif

#define TURBO_REQUIRES_IMPL(...) turbo::check_requires<__VA_ARGS__>
#define TURBO_REQUIRES(...) TURBO_REQUIRES_IMPL(__VA_ARGS__) = 0

#define TURBO_EITHER_IMPL(...) turbo::check_either<__VA_ARGS__>
#define TURBO_EITHER(...) TURBO_EITHER_IMPL(__VA_ARGS__) = 0

#define TURBO_DISALLOW_IMPL(...) turbo::check_disallow<__VA_ARGS__>
#define TURBO_DISALLOW(...) TURBO_DISALLOW_IMPL(__VA_ARGS__) = 0

#define TURBO_DISALLOW_ONE_IMPL(...) turbo::check_disallow_one<__VA_ARGS__>
#define TURBO_DISALLOW_ONE(...) TURBO_DISALLOW_ONE_IMPL(__VA_ARGS__) = 0

    // For backward compatibility
    template<class... C>
    using check_concept = check_requires<C...>;

    /**************
     * all_scalar *
     **************/

    template<class... Args>
    struct all_scalar : std::conjunction<std::is_scalar<Args>...> {
    };

    struct identity {
        template<class T>
        T &&operator()(T &&x) const {
            return std::forward<T>(x);
        }
    };

    /*************************
     * select implementation *
     *************************/

    template<class B, class T1, class T2, TURBO_REQUIRES(all_scalar<B, T1, T2>)>
    inline std::common_type_t<T1, T2> select(const B &cond, const T1 &v1, const T2 &v2) noexcept {
        return cond ? v1 : v2;
    }

    // to avoid useless casts (see https://github.com/nlohmann/json/issues/2893#issuecomment-889152324)
    template<typename T, typename U, std::enable_if_t<!std::is_same<T, U>::value, int> = 0>
    constexpr T conditional_static_cast(U value) {
        return static_cast<T>(value);
    }

    template<typename T, typename U, std::enable_if_t<std::is_same<T, U>::value, int> = 0>
    constexpr T conditional_static_cast(U value) {
        return value;
    }

    template<typename T>
    struct is_char : std::false_type {
    };
    template<>
    struct is_char<char> : std::true_type {
    };

    struct compile_string {
    };

    template<typename S>
    struct is_compile_string : std::is_base_of<compile_string, S> {
    };


    template<typename Char, TURBO_ENABLE_IF(is_char<Char>::value)>
    TURBO_FORCE_INLINE auto to_string_view(const Char *s) -> std::basic_string_view<Char> {
        return s;
    }

    template<typename Char, typename Traits, typename Alloc>
    inline auto to_string_view(const std::basic_string<Char, Traits, Alloc> &s)
    -> std::basic_string_view<Char> {
        return s;
    }

    template<typename Char>
    constexpr auto to_string_view(std::basic_string_view<Char> s)
    -> std::basic_string_view<Char> {
        return s;
    }

    template<typename S, TURBO_ENABLE_IF(is_compile_string<S>::value)>
    constexpr auto to_string_view(const S &s)
    -> std::basic_string_view<typename S::char_type> {
        return std::basic_string_view<typename S::char_type>(s);
    }

    void to_string_view(...);

    // Specifies whether S is a string type convertible to std::basic_string_view.
    // It should be a constexpr function but MSVC 2017 fails to compile it in
    // enable_if and MSVC 2015 fails to compile it as an alias template.
    // ADL is intentionally disabled as to_string_view is not an extension point.
    template<typename S>
    struct is_string
            : std::is_class<decltype(to_string_view(std::declval<S>()))> {
    };

    // range
    namespace meta_detail {
#  define FMT_DECLTYPE_RETURN(val)  \
    ->decltype(val) { return val; } \
    static_assert(                  \
        true, "")  // This makes it so that a semicolon is required after the
        // macro, which helps clang-format handle the formatting.
        // Member function overload

        // Returns true if T has a std::string-like interface, like std::string_view.
        template<typename T>
        class is_std_string_like {
            template<typename U>
            static auto check(U *p)
            -> decltype((void) p->find('a'), p->length(), (void) p->data(), int());

            template<typename>
            static void check(...);

        public:
            static constexpr const bool value =
                    is_string<T>::value ||
                    std::is_convertible<T, std::basic_string_view<char>>::value ||
                    !std::is_void<decltype(check<T>(nullptr))>::value;
        };

        template<typename Char>
        struct is_std_string_like<std::basic_string_view<Char>> : std::true_type {
        };

    }  // namespace meta_detail

    // C array overload
    template<typename T, std::size_t N>
    auto range_begin(const T (&arr)[N]) -> const T * {
        return arr;
    }

    template<typename T, std::size_t N>
    auto range_end(const T (&arr)[N]) -> const T * {
        return arr + N;
    }

    template<typename T, typename Enable = void>
    struct has_member_fn_begin_end_t : std::false_type {
    };

    template<typename T>
    struct has_member_fn_begin_end_t<T, std::void_t<decltype(std::declval<T>().begin()),
            decltype(std::declval<T>().end())>>
            : std::true_type {
    };


    template<typename T>
    auto range_begin(T &&rng) FMT_DECLTYPE_RETURN(static_cast<T &&>(rng).begin());

    template<typename T>
    auto range_end(T &&rng) FMT_DECLTYPE_RETURN(static_cast<T &&>(rng).end());

    // ADL overload. Only participates in overload resolution if member functions
    // are not found.
    template<typename T>
    auto range_begin(T &&rng)
    -> std::enable_if_t<!has_member_fn_begin_end_t<T &&>::value,
            decltype(begin(static_cast<T &&>(rng)))> {
        return begin(static_cast<T &&>(rng));
    }

    template<typename T>
    auto range_end(T &&rng) -> std::enable_if_t<!has_member_fn_begin_end_t<T &&>::value,
            decltype(end(static_cast<T &&>(rng)))> {
        return end(static_cast<T &&>(rng));
    }

    template<typename T, typename Enable = void>
    struct has_const_begin_end : std::false_type {
    };

    template<typename T>
    struct has_const_begin_end<
            T,
            std::void_t<
                    decltype(range_begin(std::declval<const turbo::remove_cvref_t<T> &>())),
                    decltype(range_end(std::declval<const turbo::remove_cvref_t<T> &>()))>>
            : std::true_type {
    };

    template<typename T, typename Enable = void>
    struct has_mutable_begin_end : std::false_type {
    };

    template<typename T>
    struct has_mutable_begin_end<
            T, std::void_t<decltype(range_begin(std::declval<T>())),
                    decltype(range_end(std::declval<T>())),
                    // the extra int here is because older versions of MSVC don't
                    // SFINAE properly unless there are distinct types
                    int>> : std::true_type {
    };


    template<typename T, typename _ = void>
    struct is_range : std::false_type {
    };

    template<typename T>
    struct is_range<T, void>
            : std::integral_constant<bool, (has_const_begin_end<T>::value ||
                                            has_mutable_begin_end<T>::value)> {
    };


    template<typename T, typename Char>
    struct is_range_printable {
        static constexpr const bool value =
                is_range<T>::value && !meta_detail::is_std_string_like<T>::value &&
                !std::is_convertible<T, std::basic_string<Char>>::value &&
                !std::is_convertible<T, std::basic_string_view<Char>>::value;
    };
#  undef FMT_DECLTYPE_RETURN

    template<class T>
    struct is_atomical : std::integral_constant<bool, (std::is_integral<T>::value ||
            std::is_floating_point<T>::value)> {
    };
    template<class T>
    struct is_atomical<const T> : is_atomical<T> {
    };
    template<class T>
    struct is_atomical<volatile T> : is_atomical<T> {
    };
    template<class T>
    struct is_atomical<const volatile T> : is_atomical<T> {
    };

    template <typename T>
    struct CachelineAligned {
        alignas (2*TURBO_CACHE_LINE_SIZE) T data;
    };

    template<bool...>
    struct bool_pack;

    template<bool... bs>
    using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

    template <typename T, typename... Ts>
    using all_same = all_true<std::is_same_v<T, Ts>...>;

    template <typename T, typename... Ts>
    constexpr bool all_same_v = all_same<T, Ts...>::value;


}  // namespace turbo

#include "turbo/meta/internal/traits.h"

//////
// TURBO_DECLVAL(T)
//
// This macro works like std::declval<T>() but does the same thing in a way
// that does not require instantiating a function template.
//
// Use this macro instead of std::declval<T>() in places that are widely
// instantiated to reduce compile-time overhead of instantiating function
// templates.
//
// Note that, like std::declval<T>(), this macro can only be used in
// unevaluated contexts.
//
// There are some small differences between this macro and std::declval<T>().
// - This macro results in a value of type 'T' instead of 'T&&'.
// - This macro requires the type T to be a complete type at the
// point of use.
// If this is a problem then use TURBO_DECLVAL(T&&) instead, or if T might
// be 'void', then use TURBO_DECLVAL(std::add_rvalue_reference_t<T>).
//
#define TURBO_DECLVAL(...) static_cast<__VA_ARGS__ (*)() noexcept>(nullptr)()

#endif  // TURBO_META_TYPE_TRAITS_H_
