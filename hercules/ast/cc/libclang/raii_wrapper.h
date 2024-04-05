// Copyright 2024 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#pragma once

#include <cstring>
#include <type_traits>
#include <utility>

#include <clang-c/Index.h>
#include <collie/type_safe/optional.h>

#include <hercules/ast/cc/detail/assert.h>

namespace hercules::ccast {
    namespace detail {
        template<typename T, class Deleter>
        class raii_wrapper : Deleter {
            static_assert(std::is_pointer<T>::value, "");

        public:
            raii_wrapper() noexcept: obj_(nullptr) {}

            explicit raii_wrapper(T obj) noexcept: obj_(obj) {
                DEBUG_ASSERT(obj_, detail::assert_handler{});
            }

            raii_wrapper(raii_wrapper &&other) noexcept: obj_(other.obj_) {
                DEBUG_ASSERT(obj_, detail::assert_handler{});
                other.obj_ = nullptr;
            }

            ~raii_wrapper() noexcept {
                if (obj_)
                    static_cast<Deleter &>(*this)(obj_);
            }

            raii_wrapper &operator=(raii_wrapper &&other) noexcept {
                raii_wrapper tmp(std::move(other));
                swap(*this, tmp);
                return *this;
            }

            friend void swap(raii_wrapper &a, raii_wrapper &b) noexcept {
                std::swap(a.obj_, b.obj_);
            }

            T get() const noexcept {
                DEBUG_ASSERT(obj_, detail::assert_handler{});
                return obj_;
            }

        private:
            T obj_;
        };

        struct cxindex_deleter {
            void operator()(CXIndex idx) noexcept {
                clang_disposeIndex(idx);
            }
        };

        using cxindex = raii_wrapper<CXIndex, cxindex_deleter>;

        struct cxtranslation_unit_deleter {
            void operator()(CXTranslationUnit unit) noexcept {
                clang_disposeTranslationUnit(unit);
            }
        };

        using cxtranslation_unit = raii_wrapper<CXTranslationUnit, cxtranslation_unit_deleter>;

        class cxstring {
        public:
            explicit cxstring(CXString str) noexcept: str_(string(str)) {}

            cxstring(cxstring &&other) noexcept: str_(other.str_) {
                other.str_.reset();
            }

            cxstring &operator=(cxstring &&other) noexcept {
                if (str_)
                    clang_disposeString(str_.value().str);
                str_ = other.str_;
                other.str_.reset();
                return *this;
            }

            ~cxstring() noexcept {
                if (str_)
                    clang_disposeString(str_.value().str);
            }

            const char *c_str() const noexcept {
                return str_ ? str_.value().c_str : "";
            }

            std::string std_str() const noexcept {
                return c_str();
            }

            char operator[](std::size_t i) const noexcept {
                return c_str()[i];
            }

            std::size_t length() const noexcept {
                return str_ ? str_.value().length : 0u;
            }

            bool empty() const noexcept {
                return length() == 0u;
            }

        private:
            struct string {
                CXString str;
                const char *c_str;
                std::size_t length;

                explicit string(CXString str)
                        : str(std::move(str)), c_str(clang_getCString(str)), length(std::strlen(c_str)) {}
            };

            collie::ts::optional<string> str_;
        };

        inline bool operator==(const cxstring &a, const cxstring &b) noexcept {
            return std::strcmp(a.c_str(), b.c_str()) == 0;
        }

        inline bool operator==(const cxstring &a, const char *str) noexcept {
            return std::strcmp(a.c_str(), str) == 0;
        }

        inline bool operator==(const char *str, const cxstring &b) noexcept {
            return std::strcmp(str, b.c_str()) == 0;
        }

        inline bool operator!=(const cxstring &a, const cxstring &b) noexcept {
            return !(a == b);
        }

        inline bool operator!=(const cxstring &a, const char *str) noexcept {
            return !(a == str);
        }

        inline bool operator!=(const char *str, const cxstring &b) noexcept {
            return !(str == b);
        }
    } // namespace detail
} // namespace hercules::ccast
