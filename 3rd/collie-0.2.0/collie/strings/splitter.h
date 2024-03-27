// Copyright 2024 The Elastic-AI Authors.
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

#pragma once

#include <cstdlib>
#include <cstdint>
#include <climits>
#include <string_view>

// It's common to encode data into strings separated by special characters
// and decode them back, but functions such as `split_string' has to modify
// the input string, which is bad. If we parse the string from scratch, the
// code will be filled with pointer operations and obscure to understand.
//
// What we want is:
// - Scan the string once: just do simple things efficiently.
// - Do not modify input string: Changing input is bad, it may bring hidden
//   bugs, concurrency issues and non-const propagations.
// - Split the string in-place without additional buffer/array.
//
// StringSplitter does meet these requirements.
// Usage:
//     const char* the_string_to_split = ...;
//     for (StringSplitter s(the_string_to_split, '\t'); s; ++s) {
//         printf("%*s\n", s.length(), s.field());    
//     }
// 
// "s" behaves as an iterator and evaluates to true before ending.
// "s.field()" and "s.length()" are address and length of current field
// respectively. Notice that "s.field()" may not end with '\0' because
// we don't modify input. You can copy the field to a dedicated buffer
// or apply a function supporting length.

namespace collie {

    enum EmptyFieldAction {
        SKIP_EMPTY_FIELD,
        ALLOW_EMPTY_FIELD
    };

    // Split a string with one character
    class StringSplitter {
    public:
        // Split `input' with `separator'. If `action' is SKIP_EMPTY_FIELD, zero-
        // length() field() will be skipped.
        inline StringSplitter(const char *input, char separator,
                              EmptyFieldAction action = SKIP_EMPTY_FIELD);

        // Allows containing embedded '\0' characters and separator can be '\0',
        // if str_end is not nullptr.
        inline StringSplitter(const char *str_begin, const char *str_end,
                              char separator,
                              EmptyFieldAction action = SKIP_EMPTY_FIELD);

        // Allows containing embedded '\0' characters and separator can be '\0',
        inline StringSplitter(const std::string_view &input, char separator,
                              EmptyFieldAction action = SKIP_EMPTY_FIELD);

        // Move splitter forward.
        inline StringSplitter &operator++();

        inline StringSplitter operator++(int);

        // True iff field() is valid.
        inline operator const void *() const;

        // Beginning address and length of the field. *(field() + length()) may
        // not be '\0' because we don't modify `input'.
        inline const char *field() const;

        inline size_t length() const;

        inline std::string_view field_sp() const;

        // Cast field to specific type, and write the value into `pv'.
        // Returns 0 on success, -1 otherwise.
        // NOTE: If separator is a digit, casting functions always return -1.
        inline int to_int8(int8_t *pv) const;

        inline int to_uint8(uint8_t *pv) const;

        inline int to_int(int *pv) const;

        inline int to_uint(unsigned int *pv) const;

        inline int to_long(long *pv) const;

        inline int to_ulong(unsigned long *pv) const;

        inline int to_longlong(long long *pv) const;

        inline int to_ulonglong(unsigned long long *pv) const;

        inline int to_float(float *pv) const;

        inline int to_double(double *pv) const;

    private:
        inline bool not_end(const char *p) const;

        inline void init();

        const char *_head;
        const char *_tail;
        const char *_str_tail;
        const char _sep;
        const EmptyFieldAction _empty_field_action;
    };

    // Split a string with one of the separators
    class StringMultiSplitter {
    public:
        // Split `input' with one character of `separators'. If `action' is
        // SKIP_EMPTY_FIELD, zero-length() field() will be skipped.
        // NOTE: This utility stores pointer of `separators' directly rather than
        //       copying the content because this utility is intended to be used
        //       in ad-hoc manner where lifetime of `separators' is generally
        //       longer than this utility.
        inline StringMultiSplitter(const char *input, const char *separators,
                                   EmptyFieldAction action = SKIP_EMPTY_FIELD);

        // Allows containing embedded '\0' characters if str_end is not nullptr.
        // NOTE: `separators` cannot contain embedded '\0' character.
        inline StringMultiSplitter(const char *str_begin, const char *str_end,
                                   const char *separators,
                                   EmptyFieldAction action = SKIP_EMPTY_FIELD);

        // Move splitter forward.
        inline StringMultiSplitter &operator++();

        inline StringMultiSplitter operator++(int);

        // True iff field() is valid.
        inline operator const void *() const;

        // Beginning address and length of the field. *(field() + length()) may
        // not be '\0' because we don't modify `input'.
        inline const char *field() const;

        inline size_t length() const;

        inline std::string_view field_sp() const;

        // Cast field to specific type, and write the value into `pv'.
        // Returns 0 on success, -1 otherwise.
        // NOTE: If separators contains digit, casting functions always return -1.
        inline int to_int8(int8_t *pv) const;

        inline int to_uint8(uint8_t *pv) const;

        inline int to_int(int *pv) const;

        inline int to_uint(unsigned int *pv) const;

        inline int to_long(long *pv) const;

        inline int to_ulong(unsigned long *pv) const;

        inline int to_longlong(long long *pv) const;

        inline int to_ulonglong(unsigned long long *pv) const;

        inline int to_float(float *pv) const;

        inline int to_double(double *pv) const;

    private:
        inline bool is_sep(char c) const;

        inline bool not_end(const char *p) const;

        inline void init();

        const char *_head;
        const char *_tail;
        const char *_str_tail;
        const char *const _seps;
        const EmptyFieldAction _empty_field_action;
    };

// Split query in the format according to the given delimiters.
// This class can also handle some exceptional cases.
// 1. consecutive pair_delimiter are omitted, for example,
//    suppose key_value_delimiter is '=' and pair_delimiter
//    is '&', then 'k1=v1&&&k2=v2' is normalized to 'k1=k2&k2=v2'.
// 2. key or value can be empty or both can be empty.
// 3. consecutive key_value_delimiter are not omitted, for example,
//    suppose input is 'k1===v2' and key_value_delimiter is '=', then
//    key() returns 'k1', value() returns '==v2'.
    class KeyValuePairsSplitter {
    public:
        inline KeyValuePairsSplitter(const char *str_begin,
                                     const char *str_end,
                                     char pair_delimiter,
                                     char key_value_delimiter)
                : _sp(str_begin, str_end, pair_delimiter), _delim_pos(std::string_view::npos),
                  _key_value_delim(key_value_delimiter) {
            UpdateDelimiterPosition();
        }

        inline KeyValuePairsSplitter(const char *str_begin,
                                     char pair_delimiter,
                                     char key_value_delimiter)
                : KeyValuePairsSplitter(str_begin, nullptr,
                                        pair_delimiter, key_value_delimiter) {}

        inline KeyValuePairsSplitter(const std::string_view &sp,
                                     char pair_delimiter,
                                     char key_value_delimiter)
                : KeyValuePairsSplitter(sp.begin(), sp.end(),
                                        pair_delimiter, key_value_delimiter) {}

        inline std::string_view key() {
            return key_and_value().substr(0, _delim_pos);
        }

        inline std::string_view value() {
            return key_and_value().substr(_delim_pos + 1);
        }

        // Get the current value of key and value
        // in the format of "key=value"
        inline std::string_view key_and_value() {
            return std::string_view(_sp.field(), _sp.length());
        }

        // Move splitter forward.
        inline KeyValuePairsSplitter &operator++() {
            ++_sp;
            UpdateDelimiterPosition();
            return *this;
        }

        inline KeyValuePairsSplitter operator++(int) {
            KeyValuePairsSplitter tmp = *this;
            operator++();
            return tmp;
        }

        inline operator const void *() const { return _sp; }

    private:
        inline void UpdateDelimiterPosition();

    private:
        StringSplitter _sp;
        std::string_view::size_type _delim_pos;
        const char _key_value_delim;
    };

    inline StringSplitter::StringSplitter(const char *str_begin,
                                   const char *str_end,
                                   const char sep,
                                   EmptyFieldAction action)
            : _head(str_begin), _str_tail(str_end), _sep(sep), _empty_field_action(action) {
        init();
    }

    inline StringSplitter::StringSplitter(const char *str, char sep,
                                   EmptyFieldAction action)
            : StringSplitter(str, nullptr, sep, action) {}

    inline StringSplitter::StringSplitter(const std::string_view &input, char sep,
                                   EmptyFieldAction action)
            : StringSplitter(input.data(), input.data() + input.length(), sep, action) {}

    inline void StringSplitter::init() {
        // Find the starting _head and _tail.
        if (__builtin_expect(_head != nullptr, 1)) {
            if (_empty_field_action == SKIP_EMPTY_FIELD) {
                for (; _sep == *_head && not_end(_head); ++_head) {}
            }
            for (_tail = _head; *_tail != _sep && not_end(_tail); ++_tail) {}
        } else {
            _tail = nullptr;
        }
    }

    inline StringSplitter &StringSplitter::operator++() {
        if (__builtin_expect(_tail != nullptr, 1)) {
            if (not_end(_tail)) {
                ++_tail;
                if (_empty_field_action == SKIP_EMPTY_FIELD) {
                    for (; _sep == *_tail && not_end(_tail); ++_tail) {}
                }
            }
            _head = _tail;
            for (; *_tail != _sep && not_end(_tail); ++_tail) {}
        }
        return *this;
    }

    inline StringSplitter StringSplitter::operator++(int) {
        StringSplitter tmp = *this;
        operator++();
        return tmp;
    }

    inline StringSplitter::operator const void *() const {
        return (_head != nullptr && not_end(_head)) ? _head : nullptr;
    }

    inline const char *StringSplitter::field() const {
        return _head;
    }

    size_t StringSplitter::length() const {
        return static_cast<size_t>(_tail - _head);
    }

    inline std::string_view StringSplitter::field_sp() const {
        return std::string_view(field(), length());
    }

    inline bool StringSplitter::not_end(const char *p) const {
        return (_str_tail == nullptr) ? *p : (p != _str_tail);
    }

    inline int StringSplitter::to_int8(int8_t *pv) const {
        long v = 0;
        if (to_long(&v) == 0 && v >= -128 && v <= 127) {
            *pv = (int8_t) v;
            return 0;
        }
        return -1;
    }

    inline int StringSplitter::to_uint8(uint8_t *pv) const {
        unsigned long v = 0;
        if (to_ulong(&v) == 0 && v <= 255) {
            *pv = (uint8_t) v;
            return 0;
        }
        return -1;
    }

    inline int StringSplitter::to_int(int *pv) const {
        long v = 0;
        if (to_long(&v) == 0 && v >= INT_MIN && v <= INT_MAX) {
            *pv = (int) v;
            return 0;
        }
        return -1;
    }

    inline int StringSplitter::to_uint(unsigned int *pv) const {
        unsigned long v = 0;
        if (to_ulong(&v) == 0 && v <= UINT_MAX) {
            *pv = (unsigned int) v;
            return 0;
        }
        return -1;
    }

    inline int StringSplitter::to_long(long *pv) const {
        char *endptr = nullptr;
        *pv = strtol(field(), &endptr, 10);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringSplitter::to_ulong(unsigned long *pv) const {
        char *endptr = nullptr;
        *pv = strtoul(field(), &endptr, 10);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringSplitter::to_longlong(long long *pv) const {
        char *endptr = nullptr;
        *pv = strtoll(field(), &endptr, 10);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringSplitter::to_ulonglong(unsigned long long *pv) const {
        char *endptr = nullptr;
        *pv = strtoull(field(), &endptr, 10);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringSplitter::to_float(float *pv) const {
        char *endptr = nullptr;
        *pv = strtof(field(), &endptr);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringSplitter::to_double(double *pv) const {
        char *endptr = nullptr;
        *pv = strtod(field(), &endptr);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline StringMultiSplitter::StringMultiSplitter(
            const char *str, const char *seps, EmptyFieldAction action)
            : _head(str), _str_tail(nullptr), _seps(seps), _empty_field_action(action) {
        init();
    }

    inline StringMultiSplitter::StringMultiSplitter(
            const char *str_begin, const char *str_end,
            const char *seps, EmptyFieldAction action)
            : _head(str_begin), _str_tail(str_end), _seps(seps), _empty_field_action(action) {
        init();
    }

    inline void StringMultiSplitter::init() {
        if (__builtin_expect(_head != nullptr, 1)) {
            if (_empty_field_action == SKIP_EMPTY_FIELD) {
                for (; is_sep(*_head) && not_end(_head); ++_head) {}
            }
            for (_tail = _head; !is_sep(*_tail) && not_end(_tail); ++_tail) {}
        } else {
            _tail = nullptr;
        }
    }

    inline StringMultiSplitter &StringMultiSplitter::operator++() {
        if (__builtin_expect(_tail != nullptr, 1)) {
            if (not_end(_tail)) {
                ++_tail;
                if (_empty_field_action == SKIP_EMPTY_FIELD) {
                    for (; is_sep(*_tail) && not_end(_tail); ++_tail) {}
                }
            }
            _head = _tail;
            for (; !is_sep(*_tail) && not_end(_tail); ++_tail) {}
        }
        return *this;
    }

    inline StringMultiSplitter StringMultiSplitter::operator++(int) {
        StringMultiSplitter tmp = *this;
        operator++();
        return tmp;
    }

    inline bool StringMultiSplitter::is_sep(char c) const {
        for (const char *p = _seps; *p != '\0'; ++p) {
            if (c == *p) {
                return true;
            }
        }
        return false;
    }

    inline StringMultiSplitter::operator const void *() const {
        return (_head != nullptr && not_end(_head)) ? _head : nullptr;
    }

    inline const char *StringMultiSplitter::field() const {
        return _head;
    }

    inline size_t StringMultiSplitter::length() const {
        return static_cast<size_t>(_tail - _head);
    }

    inline std::string_view StringMultiSplitter::field_sp() const {
        return std::string_view(field(), length());
    }

    inline bool StringMultiSplitter::not_end(const char *p) const {
        return (_str_tail == nullptr) ? *p : (p != _str_tail);
    }

    inline int StringMultiSplitter::to_int8(int8_t *pv) const {
        long v = 0;
        if (to_long(&v) == 0 && v >= -128 && v <= 127) {
            *pv = (int8_t) v;
            return 0;
        }
        return -1;
    }

    inline int StringMultiSplitter::to_uint8(uint8_t *pv) const {
        unsigned long v = 0;
        if (to_ulong(&v) == 0 && v <= 255) {
            *pv = (uint8_t) v;
            return 0;
        }
        return -1;
    }

    inline int StringMultiSplitter::to_int(int *pv) const {
        long v = 0;
        if (to_long(&v) == 0 && v >= INT_MIN && v <= INT_MAX) {
            *pv = (int) v;
            return 0;
        }
        return -1;
    }

    inline int StringMultiSplitter::to_uint(unsigned int *pv) const {
        unsigned long v = 0;
        if (to_ulong(&v) == 0 && v <= UINT_MAX) {
            *pv = (unsigned int) v;
            return 0;
        }
        return -1;
    }

    inline int StringMultiSplitter::to_long(long *pv) const {
        char *endptr = nullptr;
        *pv = strtol(field(), &endptr, 10);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringMultiSplitter::to_ulong(unsigned long *pv) const {
        char *endptr = nullptr;
        *pv = strtoul(field(), &endptr, 10);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringMultiSplitter::to_longlong(long long *pv) const {
        char *endptr = nullptr;
        *pv = strtoll(field(), &endptr, 10);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringMultiSplitter::to_ulonglong(unsigned long long *pv) const {
        char *endptr = nullptr;
        *pv = strtoull(field(), &endptr, 10);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringMultiSplitter::to_float(float *pv) const {
        char *endptr = nullptr;
        *pv = strtof(field(), &endptr);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline int StringMultiSplitter::to_double(double *pv) const {
        char *endptr = nullptr;
        *pv = strtod(field(), &endptr);
        return (endptr == field() + length()) ? 0 : -1;
    }

    inline void KeyValuePairsSplitter::UpdateDelimiterPosition() {
        const std::string_view key_value_pair(key_and_value());
        _delim_pos = key_value_pair.find(_key_value_delim);
        if (_delim_pos == std::string_view::npos) {
            _delim_pos = key_value_pair.length();
        }
    }

}  // namespace collie

