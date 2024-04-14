// Formatting library for C++ - formatters for standard library types
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_STD_H_
#define FMT_STD_H_

#include <cstdlib>
#include <exception>
#include <memory>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <optional>
#include <variant>
#include <list>
#include <vector>
#include <initializer_list>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "turbo/strings/match.h"
#include "ostream.h"

// GCC 4 does not support TURBO_HAVE_INCLUDE.
#if TURBO_HAVE_INCLUDE(<cxxabi.h>) || defined(__GLIBCXX__)

#  include <cxxabi.h>
// Android NDK with gabi++ library on some architectures does not implement
// abi::__cxa_demangle().
#  ifndef __GABIXX_CXXABI_H__
#    define FMT_HAS_ABI_CXA_DEMANGLE
#  endif
#endif

namespace turbo {
    template<typename Char>
    struct formatter<std::thread::id, Char> : basic_ostream_formatter<Char> {
    };
}

namespace turbo {
    template<typename T, typename Char>
    struct formatter<std::optional<T>, Char,
            std::enable_if_t<is_formattable<T, Char>::value>> {
    private:
        formatter<T, Char> underlying_;
        static constexpr std::basic_string_view<Char> optional =
                fmt_detail::string_literal<Char, 'o', 'p', 't', 'i', 'o', 'n', 'a', 'l',
                        '('>{};
        static constexpr std::basic_string_view<Char> none =
                fmt_detail::string_literal<Char, 'n', 'o', 'n', 'e'>{};

        template<class U>
        constexpr static auto maybe_set_debug_format(U &u, bool set)
        -> decltype(u.set_debug_format(set)) {
            u.set_debug_format(set);
        }

        template<class U>
        constexpr static void maybe_set_debug_format(U &, ...) {}

    public:
        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) {
            maybe_set_debug_format(underlying_, true);
            return underlying_.parse(ctx);
        }

        template<typename FormatContext>
        auto format(std::optional<T> const &opt, FormatContext &ctx) const
        -> decltype(ctx.out()) {
            if (!opt) return fmt_detail::write<Char>(ctx.out(), none);

            auto out = ctx.out();
            out = fmt_detail::write<Char>(out, optional);
            ctx.advance_to(out);
            out = underlying_.format(*opt, ctx);
            return fmt_detail::write(out, ')');
        }
    };

} // namespace turbo

namespace turbo {
    template<typename Char>
    struct formatter<std::monostate, Char> {
        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
            return ctx.begin();
        }

        template<typename FormatContext>
        auto format(const std::monostate &, FormatContext &ctx) const
        -> decltype(ctx.out()) {
            auto out = ctx.out();
            out = fmt_detail::write<Char>(out, "monostate");
            return out;
        }
    };

    namespace fmt_detail {

        template<typename T>
        using variant_index_sequence =
                std::make_index_sequence<std::variant_size<T>::value>;

        template<typename>
        struct is_variant_like_ : std::false_type {
        };
        template<typename... Types>
        struct is_variant_like_<std::variant<Types...>> : std::true_type {
        };

        // formattable element check.
        template<typename T, typename C>
        class is_variant_formattable_ {
            template<std::size_t... Is>
            static std::conjunction<
                    is_formattable<std::variant_alternative_t<Is, T>, C>...>
            check(std::index_sequence<Is...>);

        public:
            static constexpr const bool value =
                    decltype(check(variant_index_sequence<T>{}))::value;
        };

        template<typename Char, typename OutputIt, typename T>
        auto write_variant_alternative(OutputIt out, const T &v) -> OutputIt {
            if constexpr (is_string<T>::value)
                return write_escaped_string<Char>(out, to_string_view(v));
            else if constexpr (std::is_same_v<T, Char>)
                return write_escaped_char(out, v);
            else
                return write<Char>(out, v);
        }

    }  // namespace fmt_detail
    template<typename T>
    struct is_variant_like {
        static constexpr const bool value = fmt_detail::is_variant_like_<T>::value;
    };

    template<typename T, typename C>
    struct is_variant_formattable {
        static constexpr const bool value =
                fmt_detail::is_variant_formattable_<T, C>::value;
    };

    template<typename Variant, typename Char>
    struct formatter<
            Variant, Char,
            std::enable_if_t<std::conjunction_v<
                    is_variant_like<Variant>, is_variant_formattable<Variant, Char>>>> {
        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
            return ctx.begin();
        }

        template<typename FormatContext>
        auto format(const Variant &value, FormatContext &ctx) const
        -> decltype(ctx.out()) {
            auto out = ctx.out();

            out = fmt_detail::write<Char>(out, "variant(");
            try {
                std::visit(
                        [&](const auto &v) {
                            out = fmt_detail::write_variant_alternative<Char>(out, v);
                        },
                        value);
            } catch (const std::bad_variant_access &) {
                fmt_detail::write<Char>(out, "valueless by exception");
            }
            *out++ = ')';
            return out;
        }
    };
}  // namespace turbo

namespace turbo {

    template<typename Char>
    struct formatter<std::error_code, Char> {
        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
            return ctx.begin();
        }

        template<typename FormatContext>
        constexpr auto format(const std::error_code &ec, FormatContext &ctx) const
        -> decltype(ctx.out()) {
            auto out = ctx.out();
            out = fmt_detail::write_bytes(out, ec.category().name(), format_specs<Char>());
            out = fmt_detail::write<Char>(out, Char(':'));
            out = fmt_detail::write<Char>(out, ec.value());
            return out;
        }
    };

    template<typename T, typename Char>
    struct formatter<
            T, Char,
            typename std::enable_if<std::is_base_of<std::exception, T>::value>::type> {
    private:
        bool with_typename_ = false;

    public:
        constexpr auto parse(basic_format_parse_context<Char> &ctx)
        -> decltype(ctx.begin()) {
            auto it = ctx.begin();
            auto end = ctx.end();
            if (it == end || *it == '}') return it;
            if (*it == 't') {
                ++it;
                with_typename_ = true;
            }
            return it;
        }

        template<typename OutputIt>
        auto format(const std::exception &ex,
                    basic_format_context<OutputIt, Char> &ctx) const -> OutputIt {
            format_specs<Char> spec;
            auto out = ctx.out();
            if (!with_typename_)
                return fmt_detail::write_bytes(out, std::string_view(ex.what()), spec);

            const std::type_info &ti = typeid(ex);
#ifdef FMT_HAS_ABI_CXA_DEMANGLE
            int status = 0;
            std::size_t size = 0;
            std::unique_ptr<char, decltype(&std::free)> demangled_name_ptr(
                    abi::__cxa_demangle(ti.name(), nullptr, &size, &status), &std::free);

            std::string_view demangled_name_view;
            if (demangled_name_ptr) {
                demangled_name_view = demangled_name_ptr.get();

                // Normalization of stdlib inline namespace names.
                // libc++ inline namespaces.
                //  std::__1::*       -> std::*
                //  std::__1::__fs::* -> std::*
                // libstdc++ inline namespaces.
                //  std::__cxx11::*             -> std::*
                //  std::filesystem::__cxx11::* -> std::filesystem::*
                if (turbo::starts_with(demangled_name_view, "std::")) {
                    char *begin = demangled_name_ptr.get();
                    char *to = begin + 5;  // std::
                    for (char *from = to, *end = begin + demangled_name_view.size();
                         from < end;) {
                        // This is safe, because demangled_name is NUL-terminated.
                        if (from[0] == '_' && from[1] == '_') {
                            char *next = from + 1;
                            while (next < end && *next != ':') next++;
                            if (next[0] == ':' && next[1] == ':') {
                                from = next + 2;
                                continue;
                            }
                        }
                        *to++ = *from++;
                    }
                    demangled_name_view = {begin, turbo::to_unsigned(to - begin)};
                }
            } else {
                demangled_name_view = std::string_view(ti.name());
            }
            out = fmt_detail::write_bytes(out, demangled_name_view, spec);
#elif TURBO_MSC_VERSION
            std::string_view demangled_name_view(ti.name());
            if (demangled_name_view.starts_with("class "))
              demangled_name_view.remove_prefix(6);
            else if (demangled_name_view.starts_with("struct "))
              demangled_name_view.remove_prefix(7);
            out = fmt_detail::write_bytes(out, demangled_name_view, spec);
#else
            out = fmt_detail::write_bytes(out, std::string_view(ti.name()), spec);
#endif
            out = fmt_detail::write<Char>(out, Char(':'));
            out = fmt_detail::write<Char>(out, Char(' '));
            out = fmt_detail::write_bytes(out, std::string_view(ex.what()), spec);

            return out;
        }
    };
}  // namespace turbo

namespace turbo {

    template<typename T>
    struct is_smart_pointer : std::false_type {
    };

    template<typename T>
    struct is_smart_pointer<std::unique_ptr<T>> : std::true_type {
    };

    template<typename T>
    struct is_smart_pointer<std::shared_ptr<T>> : std::true_type {
    };

    template<typename T>
    struct is_smart_pointer<std::weak_ptr<T>> : std::true_type {
    };

    template<typename T, typename Char>
    struct formatter<T, Char,
            std::enable_if_t<is_smart_pointer<T>::value
                             && is_formattable<typename T::element_type, Char>::value>> {
        formatter<typename T::element_type, Char> underlying_;

        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
            return underlying_.parse(ctx);
        }

        template<typename FormatContext>
        auto format(const T &p, FormatContext &ctx) -> decltype(ctx.out()) {
            if (p) {
                return underlying_.format(*p, ctx);
            } else {
                return format_to(ctx.out(), "nullptr");
            }
        }
    };
}  // namespace turbo
#endif  // FMT_STD_H_
