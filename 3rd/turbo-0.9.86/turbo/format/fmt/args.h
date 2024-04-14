// Formatting library for C++ - dynamic format arguments
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_ARGS_H_
#define FMT_ARGS_H_

#include <functional>  // std::reference_wrapper
#include <memory>      // std::unique_ptr
#include <vector>

#include "core.h"

namespace turbo {

    namespace fmt_detail {

        template<typename T>
        struct is_reference_wrapper : std::false_type {
        };
        template<typename T>
        struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {
        };

        template<typename T>
        const T &unwrap(const T &v) { return v; }

        template<typename T>
        const T &unwrap(const std::reference_wrapper<T> &v) {
            return static_cast<const T &>(v);
        }

        class dynamic_arg_list {
            // Workaround for clang's -Wweak-vtables. Unlike for regular classes, for
            // templates it doesn't complain about inability to deduce single translation
            // unit for placing vtable. So storage_node_base is made a fake template.
            template<typename = void>
            struct node {
                virtual ~node() = default;

                std::unique_ptr<node<>> next;
            };

            template<typename T>
            struct typed_node : node<> {
                T value;

                template<typename Arg>
                constexpr typed_node(const Arg &arg) : value(arg) {}

                template<typename Char>
                constexpr typed_node(const std::basic_string_view<Char> &arg)
                        : value(arg.data(), arg.size()) {}
            };

            std::unique_ptr<node<>> head_;

        public:
            template<typename T, typename Arg>
            const T &push(const Arg &arg) {
                auto new_node = std::unique_ptr<typed_node<T>>(new typed_node<T>(arg));
                auto &value = new_node->value;
                new_node->next = std::move(head_);
                head_ = std::move(new_node);
                return value;
            }
        };
    }  // namespace fmt_detail

    /**
      \rst
      A dynamic version of `turbo::format_arg_store`.
      It's equipped with a storage to potentially temporary objects which lifetimes
      could be shorter than the format arguments object.

      It can be implicitly converted into `~turbo::basic_format_args` for passing
      into type-erased formatting functions such as `~turbo::vformat`.
      \endrst
     */
    template<typename Context>
    class dynamic_format_arg_store {
    private:
        using char_type = typename Context::char_type;

        template<typename T>
        struct need_copy {
            static constexpr fmt_detail::type mapped_type =
                    fmt_detail::mapped_type_constant<T, Context>::value;

            enum {
                value = !(fmt_detail::is_reference_wrapper<T>::value ||
                          std::is_same<T, std::basic_string_view<char_type>>::value ||
                          (mapped_type != fmt_detail::type::cstring_type &&
                           mapped_type != fmt_detail::type::string_type &&
                           mapped_type != fmt_detail::type::custom_type))
            };
        };

        template<typename T>
        using stored_type = std::conditional_t<
                std::is_convertible<T, std::basic_string<char_type>>::value &&
                !fmt_detail::is_reference_wrapper<T>::value,
                std::basic_string<char_type>, T>;

        // Storage of basic_format_arg must be contiguous.
        std::vector<basic_format_arg<Context>> data_;
        std::vector<fmt_detail::named_arg_info<char_type>> named_info_;

        // Storage of arguments not fitting into basic_format_arg must grow
        // without relocation because items in data_ refer to it.
        fmt_detail::dynamic_arg_list dynamic_args_;

        friend class basic_format_args<Context>;

        unsigned long long get_types() const {
            return fmt_detail::is_unpacked_bit | data_.size() |
                   (named_info_.empty()
                    ? 0ULL
                    : static_cast<unsigned long long>(fmt_detail::has_named_args_bit));
        }

        const basic_format_arg<Context> *data() const {
            return named_info_.empty() ? data_.data() : data_.data() + 1;
        }

        template<typename T>
        void emplace_arg(const T &arg) {
            data_.emplace_back(fmt_detail::make_arg<Context>(arg));
        }

        template<typename T>
        void emplace_arg(const fmt_detail::named_arg<char_type, T> &arg) {
            if (named_info_.empty()) {
                constexpr const fmt_detail::named_arg_info<char_type> *zero_ptr{nullptr};
                data_.insert(data_.begin(), {zero_ptr, 0});
            }
            data_.emplace_back(fmt_detail::make_arg<Context>(fmt_detail::unwrap(arg.value)));
            auto pop_one = [](std::vector<basic_format_arg<Context>> *data) {
                data->pop_back();
            };
            std::unique_ptr<std::vector<basic_format_arg<Context>>, decltype(pop_one)>
                    guard{&data_, pop_one};
            named_info_.push_back({arg.name, static_cast<int>(data_.size() - 2u)});
            data_[0].value_.named_args = {named_info_.data(), named_info_.size()};
            guard.release();
        }

    public:
        constexpr dynamic_format_arg_store() = default;

        /**
          \rst
          Adds an argument into the dynamic store for later passing to a formatting
          function.

          Note that custom types and string types (but not string views) are copied
          into the store dynamically allocating memory if necessary.

          **Example**::

            turbo::dynamic_format_arg_store<turbo::format_context> store;
            store.push_back(42);
            store.push_back("abc");
            store.push_back(1.5f);
            std::string result = turbo::vformat("{} and {} and {}", store);
          \endrst
        */
        template<typename T>
        void push_back(const T &arg) {
            if (fmt_detail::const_check(need_copy<T>::value))
                emplace_arg(dynamic_args_.push<stored_type<T>>(arg));
            else
                emplace_arg(fmt_detail::unwrap(arg));
        }

        /**
          \rst
          Adds a reference to the argument into the dynamic store for later passing to
          a formatting function.

          **Example**::

            turbo::dynamic_format_arg_store<turbo::format_context> store;
            char band[] = "Rolling Stones";
            store.push_back(std::cref(band));
            band[9] = 'c'; // Changing str affects the output.
            std::string result = turbo::vformat("{}", store);
            // result == "Rolling Scones"
          \endrst
        */
        template<typename T>
        void push_back(std::reference_wrapper<T> arg) {
            static_assert(
                    need_copy<T>::value,
                    "objects of built-in types and string views are always copied");
            emplace_arg(arg.get());
        }

        /**
          Adds named argument into the dynamic store for later passing to a formatting
          function. ``std::reference_wrapper`` is supported to avoid copying of the
          argument. The name is always copied into the store.
        */
        template<typename T>
        void push_back(const fmt_detail::named_arg<char_type, T> &arg) {
            const char_type *arg_name =
                    dynamic_args_.push<std::basic_string<char_type>>(arg.name).c_str();
            if (fmt_detail::const_check(need_copy<T>::value)) {
                emplace_arg(
                        turbo::arg(arg_name, dynamic_args_.push<stored_type<T>>(arg.value)));
            } else {
                emplace_arg(turbo::arg(arg_name, arg.value));
            }
        }

        /** Erase all elements from the store */
        void clear() {
            data_.clear();
            named_info_.clear();
            dynamic_args_ = fmt_detail::dynamic_arg_list();
        }

        /**
          \rst
          Reserves space to store at least *new_cap* arguments including
          *new_cap_named* named arguments.
          \endrst
        */
        void reserve(size_t new_cap, size_t new_cap_named) {
            TURBO_ASSERT(new_cap >= new_cap_named,
                         "Set of arguments includes set of named arguments");
            data_.reserve(new_cap);
            named_info_.reserve(new_cap_named);
        }
    };
}  // namespace turbo

#endif  // FMT_ARGS_H_
