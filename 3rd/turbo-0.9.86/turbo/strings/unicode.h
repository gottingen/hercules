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


#ifndef TURBO_STRINGS_UNICODE_H_
#define TURBO_STRINGS_UNICODE_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iosfwd>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "turbo/platform/port.h"
#include "turbo/strings/internal/memutil.h"
#include "turbo/strings/inlined_string.h"
#include "turbo/strings/unicode_view.h"
#include <turbo/unicode/utf.h>
#include "turbo/format/format.h"
#include "turbo/hash/hash.h"

namespace turbo {

    typedef std::basic_string<char32_t> unicode_string;

    class Unicode {
    public:
        static constexpr bool _type_is_nullable = false;

    public:
        // data holder
        using ContainerType = inlined_string_core<char32_t>;
        using self_view = unicode_view;
        using size_type = self_view::size_type;

        static constexpr size_type npos = self_view::npos;
        // types
        using value_type = char32_t;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator = value_type*;
        using const_iterator = const value_type*;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using difference_type = std::ptrdiff_t;
        using traits_type = std::char_traits<value_type>;

    public:
        // Should always true
        bool isSane() const noexcept;
        // Custom operator with string view
        operator self_view() const noexcept;
        self_view view() const noexcept;

        // C++11 21.4.2 construct/copy/destroy
        Unicode() noexcept = default;
        Unicode(const Unicode& other) = default;
        Unicode(Unicode&& other) noexcept = default;

        Unicode(const Unicode& str, size_type pos, size_type n) : Unicode(str.view().substr(pos, n)) {
        }

        // TODO: add Unicode(const Unicode& str, size_type pos)

        Unicode(const value_type* other) : Unicode(other, std::char_traits<value_type>::length(other)) {
        }

        Unicode(const value_type* other, size_type n) : data_(other, n) {
        }

        Unicode(size_type n, value_type c) : data_(n, c) {
        }

        template <typename IterType,
                typename = typename std::enable_if<!std::is_same<IterType, value_type*>::value>::type>
        Unicode(IterType first, IterType last) {
            size_type num = std::distance(first, last);
            this->reserve(num);
            while (first != last) {
                this->push_back(*first);
                ++first;
            }
        }

        // Specialization for const char*, const char*
        explicit Unicode(const value_type* b, const value_type* e) : data_(b, e - b) {
        }

        // Nonstandard constructor
        explicit Unicode(const ContainerType& other) : data_(other) {
        }
        explicit Unicode(ContainerType&& other) noexcept : data_(std::move(other)) {
        }

        Unicode(self_view other) : data_(other.data(), other.size(), other.category()) {
        }

        // Construction from initialization list
        Unicode(std::initializer_list<value_type> il) : Unicode(il.begin(), il.size()) {
        }

        // Copy assignment
        Unicode& operator=(const Unicode& other);

        // Move assignment
        Unicode& operator=(Unicode&& other) noexcept;

        Unicode& operator=(const value_type* other) {
            return other ? assign(other, std::char_traits<value_type>::length(other)) : *this;
        }

        Unicode& operator=(value_type other) {
            return operator=(Unicode(1, other));
        }

        Unicode& operator=(std::initializer_list<value_type> il) {
            return assign(il.begin(), il.size());
        }

        explicit operator std::basic_string_view<value_type, std::char_traits<value_type>>() const noexcept;

        // Nonstandard assignment
        Unicode& operator=(self_view other) {
            return assign(other);
        }

        // C++11 21.4.3 iterators:
        iterator begin();
        const_iterator begin() const noexcept;
        const_iterator cbegin() const noexcept;

        iterator end();
        const_iterator end() const noexcept;
        const_iterator cend() const noexcept;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const noexcept;
        const_reverse_iterator crbegin() const noexcept;

        reverse_iterator rend();
        const_reverse_iterator rend() const noexcept;
        const_reverse_iterator crend() const noexcept;

        // Added by C++11
        // C++11 21.4.5, element access:
        const value_type& front() const noexcept;
        const value_type& back() const noexcept;
        value_type& front();
        value_type& back();
        void pop_back();

        // C++11 21.4.4 capacity:
        size_t size() const noexcept;

        size_t length() const noexcept;

        size_type max_size() const noexcept {
            return std::numeric_limits<size_type>::max();
        }

        void resize(size_type n, value_type c = '\0');
        void resizeNoInit(size_type n);  // custom resize

        int64_t capacity() const noexcept;

        void reserve(size_type res_arg = 0);

        void shrink_to_fit();

        void clear();

        bool empty() const noexcept;

        // C++11 21.4.5 element access:
        const_reference operator[](size_type pos) const noexcept;
        reference operator[](size_type pos);

        const_reference at(size_type n) const;
        reference at(size_type n);

        // C++11 21.4.6 modifiers:
        Unicode& operator+=(value_type c);
        Unicode& operator+=(std::initializer_list<value_type> il);
        // Nonstandard alternative for:
        //   Unicode& operator+=(const Unicode& str);
        //   Unicode& operator+=(const value_type* s);
        Unicode& operator+=(self_view s);

        Unicode& append(self_view str);
        Unicode& append(self_view str, size_type pos, size_type n);
        Unicode& append(size_type n, value_type c);

        template <class InputIterator>
        Unicode& append(InputIterator first, InputIterator last) {
            size_type num = std::distance(first, last);
            this->reserve(size() + num);
            while (first != last) {
                this->push_back(*first);
                ++first;
            }
            return *this;
        }

        Unicode& append(std::initializer_list<value_type> il) {
            return append(il.begin(), il.end());
        }

        void push_back(value_type c);  // primitive

        Unicode& assign(const Unicode& str);
        Unicode& assign(Unicode&& str);
        Unicode& assign(const Unicode& str, size_type pos, size_type n);
        Unicode& assign(const value_type* s, size_type n);
        Unicode& assign(const value_type* s);

        template <class ItOrLength, class ItOrChar>
        Unicode& assign(ItOrLength first_or_n, ItOrChar last_or_c) {
            return assign(Unicode(first_or_n, last_or_c));
        }

        Unicode& assign(std::initializer_list<value_type> il) {
            return assign(il.begin(), il.size());
        }

        // Nonstandard assignment:
        Unicode& assign(self_view sv);

        Unicode& insert(size_type pos1, const Unicode& str);
        Unicode& insert(size_type pos1, const Unicode& str, size_type pos2, size_type n);
        Unicode& insert(size_type pos, const value_type* s);
        Unicode& insert(size_type pos, const value_type* s, size_type n);
        Unicode& insert(size_type pos, size_type n, value_type c);

        Unicode& insert(size_type pos, self_view s);  // Nonstandard

        // TODO: insert by iterator

        Unicode& erase(size_type pos = 0, size_type n = npos);

        // TODO: erase by iterator

        // Replaces at most n1 chars of *this, starting with pos1 with the
        // content of s
        Unicode& replace(size_type pos1, size_type n1, self_view s);

        // Replaces at most n1 chars of *this, starting with pos1,
        // with at most n2 chars of s starting with pos2
        Unicode& replace(size_type pos1, size_type n1, self_view s, size_type pos2, size_type n2);

        // Replaces at most n1 chars of *this, starting with pos, with n2
        // occurrences of c
        //
        // consolidated with
        //
        // Replaces at most n1 chars of *this, starting with pos, with at
        // most n2 chars of str.  str must have at least n2 chars.
        Unicode& replace(size_type pos, size_type n1, self_view s, size_type n2);
        Unicode& replace(size_type pos, size_type n1, size_type n2, value_type c);

        // TODO: replace by iterator

        // TODO: add copy

        // TODO: add swap

        const value_type* c_str() const noexcept;
        const value_type* data() const noexcept;

        size_type find(self_view str, size_type pos = 0) const;
        size_type find(value_type c, size_type pos = 0) const;

        size_type rfind(self_view str, size_type pos = npos) const;
        size_type rfind(value_type c, size_type pos = npos) const;

        size_type find_first_of(self_view str, size_type pos = 0) const;
        size_type find_first_of(value_type c, size_type pos = 0) const;

        size_type find_last_of(self_view str, size_type pos = npos) const;
        size_type find_last_of(value_type c, size_type pos = npos) const;

        size_type find_first_not_of(self_view str, size_type pos = 0) const;
        size_type find_first_not_of(value_type c, size_type pos = 0) const;

        size_type find_last_not_of(self_view str, size_type pos = npos) const;
        size_type find_last_not_of(value_type c, size_type pos = npos) const;

        // TODO: find by pointer

        Unicode substr(size_type pos = 0, size_type n = npos) const;

        int compare(const Unicode& other) const noexcept;

        /******************************************************************************
         * Nonstandard logical operator
         *****************************************************************************/
        inline bool operator==(self_view other) const noexcept {
            return view() == other;
        }
        inline bool operator!=(self_view other) const noexcept {
            return view() != other;
        }
        inline bool operator<(self_view other) const noexcept {
            return view() < other;
        }
        inline bool operator>(self_view other) const noexcept {
            return view() > other;
        }
        inline bool operator<=(self_view other) const noexcept {
            return view() <= other;
        }
        inline bool operator>=(self_view other) const noexcept {
            return view() >= other;
        }

        int64_t count(self_view x,
                      int64_t start = 0,
                      int64_t end = std::numeric_limits<int64_t>::max()) const noexcept;

        static Unicode concat(self_view lhs, self_view rhs);

        static Unicode concat(std::initializer_list<self_view> args);

        static Unicode repeat(self_view sv, int64_t times);

    private:

    private:
        ContainerType data_;
    };


    // Overload + operator
    inline Unicode operator+(const Unicode& lhs, const Unicode& rhs) {
        return Unicode::concat(Unicode::self_view(lhs), Unicode::self_view(rhs));
    }

    inline Unicode operator+(const char32_t* lhs, const Unicode& rhs) {
        return Unicode::concat(Unicode::self_view(lhs), Unicode::self_view(rhs));
    }

    inline Unicode operator+(const Unicode& lhs, const char32_t* rhs) {
        return Unicode::concat(Unicode::self_view(lhs), Unicode::self_view(rhs));
    }

    inline Unicode operator+(const Unicode& lhs, const unicode_view& rhs) {
        return Unicode::concat(Unicode::self_view(lhs), rhs);
    }

    inline Unicode operator+(const unicode_view& lhs, const Unicode& rhs) {
        return Unicode::concat(lhs, Unicode::self_view(rhs));
    }

    inline Unicode operator+(const Unicode& lhs, const unicode_string& rhs) {
        return Unicode::concat(Unicode::self_view(lhs), Unicode::self_view(rhs));
    }

    inline Unicode operator+(const unicode_string& lhs, const Unicode& rhs) {
        return Unicode::concat(Unicode::self_view(lhs), Unicode::self_view(rhs));
    }

    std::ostream& operator<<(std::ostream& out, const Unicode& input);

}  // namespace turbo

#endif  // TURBO_STRINGS_UNICODE_H_
