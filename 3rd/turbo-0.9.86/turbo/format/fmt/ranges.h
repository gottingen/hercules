// Formatting library for C++ - experimental range support
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.
//
// Copyright (c) 2018 - present, Remotion (Igor Schulz)
// All Rights Reserved
// {fmt} support for ranges, containers and types tuple interface.

#ifndef FMT_RANGES_H_
#define FMT_RANGES_H_

#include <initializer_list>
#include <tuple>
#include <type_traits>

#include "format.h"

namespace turbo {

    namespace fmt_detail {

        template<typename Range, typename OutputIt>
        auto copy(const Range &range, OutputIt out) -> OutputIt {
            for (auto it = range.begin(), end = range.end(); it != end; ++it)
                *out++ = *it;
            return out;
        }

        template<typename OutputIt>
        auto copy(const char *str, OutputIt out) -> OutputIt {
            while (*str) *out++ = *str++;
            return out;
        }

        template<typename OutputIt>
        auto copy(char ch, OutputIt out) -> OutputIt {
            *out++ = ch;
            return out;
        }

        template<typename OutputIt>
        auto copy(wchar_t ch, OutputIt out) -> OutputIt {
            *out++ = ch;
            return out;
        }

        template<typename T>
        class is_map {
            template<typename U>
            static auto check(U *) -> typename U::mapped_type;

            template<typename>
            static void check(...);

        public:
#ifdef FMT_FORMAT_MAP_AS_LIST  // DEPRECATED!
            static constexpr const bool value = false;
#else
            static constexpr const bool value =
                    !std::is_void<decltype(check<T>(nullptr))>::value;
#endif
        };

        template<typename T>
        class is_set {
            template<typename U>
            static auto check(U *) -> typename U::key_type;

            template<typename>
            static void check(...);

        public:
#ifdef FMT_FORMAT_SET_AS_LIST  // DEPRECATED!
            static constexpr const bool value = false;
#else
            static constexpr const bool value =
                    !std::is_void<decltype(check<T>(nullptr))>::value && !is_map<T>::value;
#endif
        };

        template<typename... Ts>
        struct conditional_helper {
        };



#if !TURBO_MSC_VERSION || TURBO_MSC_VERSION > 1800









#endif

// tuple_size and tuple_element check.
        template<typename T>
        class is_tuple_like_ {
            template<typename U>
            static auto check(U *p) -> decltype(std::tuple_size<U>::value, int());

            template<typename>
            static void check(...);

        public:
            static constexpr const bool value =
                    !std::is_void<decltype(check<T>(nullptr))>::value;
        };

// Check for integer_sequence
#if defined(__cpp_lib_integer_sequence) || TURBO_MSC_VERSION >= 1900
        template<typename T, T... N>
        using integer_sequence = std::integer_sequence<T, N...>;
        template<size_t... N> using index_sequence = std::index_sequence<N...>;
        template<size_t N> using make_index_sequence = std::make_index_sequence<N>;
#else
        template <typename T, T... N> struct integer_sequence {
          using value_type = T;

          static constexpr size_t size() { return sizeof...(N); }
        };

        template <size_t... N> using index_sequence = integer_sequence<size_t, N...>;

        template <typename T, size_t N, T... Ns>
        struct make_integer_sequence : make_integer_sequence<T, N - 1, N - 1, Ns...> {};
        template <typename T, T... Ns>
        struct make_integer_sequence<T, 0, Ns...> : integer_sequence<T, Ns...> {};

        template <size_t N>
        using make_index_sequence = make_integer_sequence<size_t, N>;
#endif

        template<typename T>
        using tuple_index_sequence = make_index_sequence<std::tuple_size<T>::value>;

        template<typename T, typename C, bool = is_tuple_like_<T>::value>
        class is_tuple_formattable_ {
        public:
            static constexpr const bool value = false;
        };

        template<typename T, typename C>
        class is_tuple_formattable_<T, C, true> {
            template<std::size_t... Is>
            static std::true_type check2(index_sequence<Is...>,
                                         integer_sequence<bool, (Is == Is)...>);

            static std::false_type check2(...);

            template<std::size_t... Is>
            static decltype(check2(
                    index_sequence<Is...>{},
                    integer_sequence<
                            bool, (is_formattable<typename std::tuple_element<Is, T>::type,
                                    C>::value)...>{})) check(index_sequence<Is...>);

        public:
            static constexpr const bool value =
                    decltype(check(tuple_index_sequence<T>{}))::value;
        };

        template<typename Tuple, typename F, size_t... Is>
        constexpr void for_each(index_sequence<Is...>, Tuple &&t, F &&f) {
            using std::get;
            // Using a free function get<Is>(Tuple) now.
            const int unused[] = {0, ((void) f(get<Is>(t)), 0)...};
            TURBO_UNUSED(unused);
        }

        template<typename Tuple, typename F>
        constexpr void for_each(Tuple &&t, F &&f) {
            for_each(tuple_index_sequence<turbo::remove_cvref_t<Tuple>>(),
                     std::forward<Tuple>(t), std::forward<F>(f));
        }

        template<typename Tuple1, typename Tuple2, typename F, size_t... Is>
        void for_each2(index_sequence<Is...>, Tuple1 &&t1, Tuple2 &&t2, F &&f) {
            using std::get;
            const int unused[] = {0, ((void) f(get<Is>(t1), get<Is>(t2)), 0)...};
            TURBO_UNUSED(unused);
        }

        template<typename Tuple1, typename Tuple2, typename F>
        void for_each2(Tuple1 &&t1, Tuple2 &&t2, F &&f) {
            for_each2(tuple_index_sequence<turbo::remove_cvref_t<Tuple1>>(),
                      std::forward<Tuple1>(t1), std::forward<Tuple2>(t2),
                      std::forward<F>(f));
        }

        namespace tuple {
            // Workaround a bug in MSVC 2019 (v140).
            template<typename Char, typename... T>
            using result_t = std::tuple<formatter<turbo::remove_cvref_t<T>, Char>...>;

            using std::get;

            template<typename Tuple, typename Char, std::size_t... Is>
            auto get_formatters(index_sequence<Is...>)
            -> result_t<Char, decltype(get<Is>(std::declval<Tuple>()))...>;
        }  // namespace tuple

#if TURBO_MSC_VERSION && TURBO_MSC_VERSION < 1920
        // Older MSVC doesn't get the reference type correctly for arrays.
        template <typename R> struct range_reference_type_impl {
          using type = decltype(*fmt_detail::range_begin(std::declval<R&>()));
        };

        template <typename T, std::size_t N> struct range_reference_type_impl<T[N]> {
          using type = T&;
        };

        template <typename T>
        using range_reference_type = typename range_reference_type_impl<T>::type;
#else
        template<typename Range>
        using range_reference_type =
                decltype(*range_begin(std::declval<Range &>()));
#endif

        // We don't use the Range's value_type for anything, but we do need the Range's
        // reference type, with cv-ref stripped.
        template<typename Range>
        using uncvref_type = turbo::remove_cvref_t<range_reference_type<Range>>;

        template<typename Formatter>
        constexpr auto maybe_set_debug_format(Formatter &f, bool set)
        -> decltype(f.set_debug_format(set)) {
            f.set_debug_format(set);
        }

        template<typename Formatter>
        constexpr void maybe_set_debug_format(Formatter &, ...) {}

        // These are not generic lambdas for compatibility with C++11.
        template<typename ParseContext>
        struct parse_empty_specs {
            template<typename Formatter>
            constexpr void operator()(Formatter &f) {
                f.parse(ctx);
                fmt_detail::maybe_set_debug_format(f, true);
            }

            ParseContext &ctx;
        };

        template<typename FormatContext>
        struct format_tuple_element {
            using char_type = typename FormatContext::char_type;

            template<typename T>
            void operator()(const formatter<T, char_type> &f, const T &v) {
                if (i > 0)
                    ctx.advance_to(fmt_detail::copy_str<char_type>(separator, ctx.out()));
                ctx.advance_to(f.format(v, ctx));
                ++i;
            }

            int i;
            FormatContext &ctx;
            std::basic_string_view<char_type> separator;
        };

    }  // namespace fmt_detail

    template<typename T>
    struct is_tuple_like {
        static constexpr const bool value =
                fmt_detail::is_tuple_like_<T>::value && !is_range<T>::value;
    };

    template<typename T, typename C>
    struct is_tuple_formattable {
        static constexpr const bool value =
                fmt_detail::is_tuple_formattable_<T, C>::value;
    };

    template<typename Tuple, typename Char>
    struct formatter<Tuple, Char,
            std::enable_if_t<turbo::is_tuple_like<Tuple>::value &&
                             turbo::is_tuple_formattable<Tuple, Char>::value>> {
    private:
        decltype(fmt_detail::tuple::get_formatters<Tuple, Char>(
                fmt_detail::tuple_index_sequence<Tuple>())) formatters_;

        std::basic_string_view<Char> separator_ = fmt_detail::string_literal<Char, ',', ' '>{};
        std::basic_string_view<Char> opening_bracket_ =
                fmt_detail::string_literal<Char, '('>{};
        std::basic_string_view<Char> closing_bracket_ =
                fmt_detail::string_literal<Char, ')'>{};

    public:
        constexpr formatter() {}

        constexpr void set_separator(std::basic_string_view<Char> sep) {
            separator_ = sep;
        }

        constexpr void set_brackets(std::basic_string_view<Char> open,
                                    std::basic_string_view<Char> close) {
            opening_bracket_ = open;
            closing_bracket_ = close;
        }

        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
            auto it = ctx.begin();
            if (it != ctx.end() && *it != '}')
                FMT_THROW(format_error("invalid format specifier"));
            fmt_detail::for_each(formatters_, fmt_detail::parse_empty_specs<ParseContext>{ctx});
            return it;
        }

        template<typename FormatContext>
        auto format(const Tuple &value, FormatContext &ctx) const
        -> decltype(ctx.out()) {
            ctx.advance_to(fmt_detail::copy_str<Char>(opening_bracket_, ctx.out()));
            fmt_detail::for_each2(
                    formatters_, value,
                    fmt_detail::format_tuple_element<FormatContext>{0, ctx, separator_});
            return fmt_detail::copy_str<Char>(closing_bracket_, ctx.out());
        }
    };

    namespace fmt_detail {
        template<typename Context>
        struct range_mapper {
            using mapper = arg_mapper<Context>;

            template<typename T,
                    TURBO_ENABLE_IF(has_formatter<turbo::remove_cvref_t<T>, Context>::value)>
            static auto map(T &&value) -> T && {
                return static_cast<T &&>(value);
            }

            template<typename T,
                    TURBO_ENABLE_IF(!has_formatter<turbo::remove_cvref_t<T>, Context>::value)>
            static auto map(T &&value)
            -> decltype(mapper().map(static_cast<T &&>(value))) {
                return mapper().map(static_cast<T &&>(value));
            }
        };

        template<typename Char, typename Element>
        using range_formatter_type =
                formatter<turbo::remove_cvref_t<decltype(range_mapper<buffer_context<Char>>{}.map(
                        std::declval<Element>()))>,
                        Char>;

        template<typename R>
        using maybe_const_range =
                std::conditional_t<has_const_begin_end<R>::value, const R, R>;

// Workaround a bug in MSVC 2015 and earlier.
#if !TURBO_MSC_VERSION || TURBO_MSC_VERSION >= 1910
        template<typename R, typename Char>
        struct is_formattable_delayed
                : is_formattable<uncvref_type<maybe_const_range<R>>, Char> {
        };
#endif
    }  // namespace fmt_detail

    template<typename T, typename Char, typename Enable = void>
    struct range_formatter;

    template<typename T, typename Char>
    struct range_formatter<
            T, Char,
            std::enable_if_t<conjunction<std::is_same<T, turbo::remove_cvref_t<T>>,
                    is_formattable<T, Char>>::value>> {
    private:
        fmt_detail::range_formatter_type<Char, T> underlying_;
        std::basic_string_view<Char> separator_ = fmt_detail::string_literal<Char, ',', ' '>{};
        std::basic_string_view<Char> opening_bracket_ =
                fmt_detail::string_literal<Char, '['>{};
        std::basic_string_view<Char> closing_bracket_ =
                fmt_detail::string_literal<Char, ']'>{};

    public:
        constexpr range_formatter() {}

        constexpr auto underlying() -> fmt_detail::range_formatter_type<Char, T> & {
            return underlying_;
        }

        constexpr void set_separator(std::basic_string_view<Char> sep) {
            separator_ = sep;
        }

        constexpr void set_brackets(std::basic_string_view<Char> open,
                                    std::basic_string_view<Char> close) {
            opening_bracket_ = open;
            closing_bracket_ = close;
        }

        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
            auto it = ctx.begin();
            auto end = ctx.end();

            if (it != end && *it == 'n') {
                set_brackets({}, {});
                ++it;
            }

            if (it != end && *it != '}') {
                if (*it != ':') FMT_THROW(format_error("invalid format specifier"));
                ++it;
            } else {
                fmt_detail::maybe_set_debug_format(underlying_, true);
            }

            ctx.advance_to(it);
            return underlying_.parse(ctx);
        }

        template<typename R, typename FormatContext>
        auto format(R &&range, FormatContext &ctx) const -> decltype(ctx.out()) {
            fmt_detail::range_mapper<buffer_context<Char>> mapper;
            auto out = ctx.out();
            out = fmt_detail::copy_str<Char>(opening_bracket_, out);
            int i = 0;
            auto it = range_begin(range);
            auto end = range_end(range);
            for (; it != end; ++it) {
                if (i > 0) out = fmt_detail::copy_str<Char>(separator_, out);
                ctx.advance_to(out);
                out = underlying_.format(mapper.map(*it), ctx);
                ++i;
            }
            out = fmt_detail::copy_str<Char>(closing_bracket_, out);
            return out;
        }
    };

    enum class range_format {
        disabled, map, set, sequence, string, debug_string
    };

    namespace fmt_detail {
        template<typename T>
        struct range_format_kind_
                : std::integral_constant<range_format,
                        std::is_same<uncvref_type<T>, T>::value
                        ? range_format::disabled
                        : is_map<T>::value ? range_format::map
                                           : is_set<T>::value ? range_format::set
                                                              : range_format::sequence> {
        };

        template<range_format K, typename R, typename Char, typename Enable = void>
        struct range_default_formatter;

        template<range_format K>
        using range_format_constant = std::integral_constant<range_format, K>;

        template<range_format K, typename R, typename Char>
        struct range_default_formatter<
                K, R, Char,
                std::enable_if_t<(K == range_format::sequence || K == range_format::map ||
                                  K == range_format::set)>> {
            using range_type = fmt_detail::maybe_const_range<R>;
            range_formatter<fmt_detail::uncvref_type<range_type>, Char> underlying_;

            constexpr range_default_formatter() { init(range_format_constant<K>()); }

            constexpr void init(range_format_constant<range_format::set>) {
                underlying_.set_brackets(fmt_detail::string_literal<Char, '{'>{},
                                         fmt_detail::string_literal<Char, '}'>{});
            }

            constexpr void init(range_format_constant<range_format::map>) {
                underlying_.set_brackets(fmt_detail::string_literal<Char, '{'>{},
                                         fmt_detail::string_literal<Char, '}'>{});
                underlying_.underlying().set_brackets({}, {});
                underlying_.underlying().set_separator(
                        fmt_detail::string_literal<Char, ':', ' '>{});
            }

            constexpr void init(range_format_constant<range_format::sequence>) {}

            template<typename ParseContext>
            constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
                return underlying_.parse(ctx);
            }

            template<typename FormatContext>
            auto format(range_type &range, FormatContext &ctx) const
            -> decltype(ctx.out()) {
                return underlying_.format(range, ctx);
            }
        };
    }  // namespace fmt_detail

    template<typename T, typename Char, typename Enable = void>
    struct range_format_kind
            : std::conditional_t<
                    is_range_printable<T, Char>::value, fmt_detail::range_format_kind_<T>,
                    std::integral_constant<range_format, range_format::disabled>> {
    };

    template<typename R, typename Char>
    struct formatter<
            R, Char,
            std::enable_if_t<conjunction<std::bool_constant<range_format_kind<R, Char>::value !=
                                                            range_format::disabled>
// Workaround a bug in MSVC 2015 and earlier.
#if !TURBO_MSC_VERSION || TURBO_MSC_VERSION >= 1910
                    ,
                    fmt_detail::is_formattable_delayed<R, Char>
#endif
            >::value>>
            : fmt_detail::range_default_formatter<range_format_kind<R, Char>::value, R,
                    Char> {
    };

    template<typename Char, typename... T>
    struct tuple_join_view : fmt_detail::view {
        const std::tuple<T...> &tuple;
        std::basic_string_view<Char> sep;

        tuple_join_view(const std::tuple<T...> &t, std::basic_string_view<Char> s)
                : tuple(t), sep{s} {}
    };

// Define FMT_TUPLE_JOIN_SPECIFIERS to enable experimental format specifiers
// support in tuple_join. It is disabled by default because of issues with
// the dynamic width and precision.
#ifndef FMT_TUPLE_JOIN_SPECIFIERS
#  define FMT_TUPLE_JOIN_SPECIFIERS 0
#endif

    template<typename Char, typename... T>
    struct formatter<tuple_join_view<Char, T...>, Char> {
        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
            return do_parse(ctx, std::integral_constant<size_t, sizeof...(T)>());
        }

        template<typename FormatContext>
        auto format(const tuple_join_view<Char, T...> &value,
                    FormatContext &ctx) const -> typename FormatContext::iterator {
            return do_format(value, ctx,
                             std::integral_constant<size_t, sizeof...(T)>());
        }

    private:
        std::tuple<formatter<typename std::decay<T>::type, Char>...> formatters_;

        template<typename ParseContext>
        constexpr auto do_parse(ParseContext &ctx,
                                std::integral_constant<size_t, 0>)
        -> decltype(ctx.begin()) {
            return ctx.begin();
        }

        template<typename ParseContext, size_t N>
        constexpr auto do_parse(ParseContext &ctx,
                                std::integral_constant<size_t, N>)
        -> decltype(ctx.begin()) {
            auto end = ctx.begin();
#if FMT_TUPLE_JOIN_SPECIFIERS
            end = std::get<sizeof...(T) - N>(formatters_).parse(ctx);
            if (N > 1) {
              auto end1 = do_parse(ctx, std::integral_constant<size_t, N - 1>());
              if (end != end1)
                FMT_THROW(format_error("incompatible format specs for tuple elements"));
            }
#endif
            return end;
        }

        template<typename FormatContext>
        auto do_format(const tuple_join_view<Char, T...> &, FormatContext &ctx,
                       std::integral_constant<size_t, 0>) const ->
        typename FormatContext::iterator {
            return ctx.out();
        }

        template<typename FormatContext, size_t N>
        auto do_format(const tuple_join_view<Char, T...> &value, FormatContext &ctx,
                       std::integral_constant<size_t, N>) const ->
        typename FormatContext::iterator {
            auto out = std::get<sizeof...(T) - N>(formatters_)
                    .format(std::get<sizeof...(T) - N>(value.tuple), ctx);
            if (N > 1) {
                out = std::copy(value.sep.begin(), value.sep.end(), out);
                ctx.advance_to(out);
                return do_format(value, ctx, std::integral_constant<size_t, N - 1>());
            }
            return out;
        }
    };

    namespace fmt_detail {
        // Check if T has an interface like a container adaptor (e.g. std::stack,
        // std::queue, std::priority_queue).
        template<typename T>
        class is_container_adaptor_like {
            template<typename U>
            static auto check(U *p) -> typename U::container_type;

            template<typename>
            static void check(...);

        public:
            static constexpr const bool value =
                    !std::is_void<decltype(check<T>(nullptr))>::value;
        };

        template<typename Container>
        struct all {
            const Container &c;

            auto begin() const -> typename Container::const_iterator { return c.begin(); }

            auto end() const -> typename Container::const_iterator { return c.end(); }
        };
    }  // namespace fmt_detail

    template<typename T, typename Char>
    struct formatter<T, Char,
            std::enable_if_t<fmt_detail::is_container_adaptor_like<T>::value>>
            : formatter<fmt_detail::all<typename T::container_type>, Char> {
        using all = fmt_detail::all<typename T::container_type>;

        template<typename FormatContext>
        auto format(const T &t, FormatContext &ctx) const -> decltype(ctx.out()) {
            struct getter : T {
                static auto get(const T &t) -> all {
                    return {t.*(&getter::c)};  // Access c through the derived class.
                }
            };
            return formatter<all>::format(getter::get(t), ctx);
        }
    };


    /**
      \rst
      Returns an object that formats `tuple` with elements separated by `sep`.

      **Example**::

        std::tuple<int, char> t = {1, 'a'};
        turbo::print("{}", turbo::join(t, ", "));
        // Output: "1, a"
      \endrst
 */
    template<typename... T>
    constexpr auto join(const std::tuple<T...> &tuple, std::string_view sep)
    -> tuple_join_view<char, T...> {
        return {tuple, sep};
    }

    template<typename... T>
    constexpr auto join(const std::tuple<T...> &tuple,
                        std::basic_string_view<wchar_t> sep)
    -> tuple_join_view<wchar_t, T...> {
        return {tuple, sep};
    }

    /**
      \rst
      Returns an object that formats `initializer_list` with elements separated by
      `sep`.

      **Example**::

        turbo::print("{}", turbo::join({1, 2, 3}, ", "));
        // Output: "1, 2, 3"
      \endrst
     */
    template<typename T>
    auto join(std::initializer_list<T> list, std::string_view sep)
    -> join_view<const T *, const T *> {
        return join(std::begin(list), std::end(list), sep);
    }

}  // namespace turbo

#endif  // FMT_RANGES_H_
