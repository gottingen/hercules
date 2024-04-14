//
// Copyright 2022 The Turbo Authors.
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
// -----------------------------------------------------------------------------
// File: ascii.h
// -----------------------------------------------------------------------------
//
// This package contains functions operating on characters and strings
// restricted to standard ASCII. These include character classification
// functions analogous to those found in the ANSI C Standard Library <ctype.h>
// header file.
//
// C++ implementations provide <ctype.h> functionality based on their
// C environment locale. In general, reliance on such a locale is not ideal, as
// the locale standard is problematic (and may not return invariant information
// for the same character set, for example). These `ascii_*()` functions are
// hard-wired for standard ASCII, much faster, and guaranteed to behave
// consistently.  They will never be overloaded, nor will their function
// signature change.
//
// `ascii_is_alnum()`, `ascii_is_alpha()`, `ascii_is_ascii()`, `ascii_is_blank()`,
// `ascii_is_cntrl()`, `ascii_is_digit()`, `ascii_is_graph()`, `ascii_is_lower()`,
// `ascii_is_print()`, `ascii_is_punct()`, `ascii_is_space()`, `ascii_is_upper()`,
// `ascii_is_xdigit()`
//   Analogous to the <ctype.h> functions with similar names, these
//   functions take an unsigned char and return a bool, based on whether the
//   character matches the condition specified.
//
//   If the input character has a numerical value greater than 127, these
//   functions return `false`.
//
// `ascii_to_lower()`, `ascii_to_upper()`
//   Analogous to the <ctype.h> functions with similar names, these functions
//   take an unsigned char and return a char.
//
//   If the input character is not an ASCII {lower,upper}-case letter (including
//   numerical values greater than 127) then the functions return the same value
//   as the input character.

#ifndef TURBO_STRINGS_ASCII_H_
#define TURBO_STRINGS_ASCII_H_

#include <algorithm>
#include <string>

#include "turbo/platform/port.h"
#include "turbo/strings/string_view.h"
#include "turbo/meta/type_traits.h"

namespace turbo {
    namespace ascii_internal {

        // Declaration for an array of bitfields holding character information.
        TURBO_DLL constexpr unsigned char kPropertyBits[256] = {
                0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0x00
                0x40, 0x68, 0x48, 0x48, 0x48, 0x48, 0x40, 0x40,
                0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0x10
                0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
                0x28, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,  // 0x20
                0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
                0x84, 0x84, 0x84, 0x84, 0x84, 0x84, 0x84, 0x84,  // 0x30
                0x84, 0x84, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
                0x10, 0x85, 0x85, 0x85, 0x85, 0x85, 0x85, 0x05,  // 0x40
                0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
                0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,  // 0x50
                0x05, 0x05, 0x05, 0x10, 0x10, 0x10, 0x10, 0x10,
                0x10, 0x85, 0x85, 0x85, 0x85, 0x85, 0x85, 0x05,  // 0x60
                0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
                0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,  // 0x70
                0x05, 0x05, 0x05, 0x10, 0x10, 0x10, 0x10, 0x40,
        };

        // Declaration for the array of characters to upper-case characters.
        TURBO_DLL constexpr char kToUpper[256] = {
                '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
                '\x08', '\x09', '\x0a', '\x0b', '\x0c', '\x0d', '\x0e', '\x0f',
                '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
                '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
                '\x20', '\x21', '\x22', '\x23', '\x24', '\x25', '\x26', '\x27',
                '\x28', '\x29', '\x2a', '\x2b', '\x2c', '\x2d', '\x2e', '\x2f',
                '\x30', '\x31', '\x32', '\x33', '\x34', '\x35', '\x36', '\x37',
                '\x38', '\x39', '\x3a', '\x3b', '\x3c', '\x3d', '\x3e', '\x3f',
                '\x40', '\x41', '\x42', '\x43', '\x44', '\x45', '\x46', '\x47',
                '\x48', '\x49', '\x4a', '\x4b', '\x4c', '\x4d', '\x4e', '\x4f',
                '\x50', '\x51', '\x52', '\x53', '\x54', '\x55', '\x56', '\x57',
                '\x58', '\x59', '\x5a', '\x5b', '\x5c', '\x5d', '\x5e', '\x5f',
                '\x60', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                'X', 'Y', 'Z', '\x7b', '\x7c', '\x7d', '\x7e', '\x7f',
                '\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87',
                '\x88', '\x89', '\x8a', '\x8b', '\x8c', '\x8d', '\x8e', '\x8f',
                '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97',
                '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9d', '\x9e', '\x9f',
                '\xa0', '\xa1', '\xa2', '\xa3', '\xa4', '\xa5', '\xa6', '\xa7',
                '\xa8', '\xa9', '\xaa', '\xab', '\xac', '\xad', '\xae', '\xaf',
                '\xb0', '\xb1', '\xb2', '\xb3', '\xb4', '\xb5', '\xb6', '\xb7',
                '\xb8', '\xb9', '\xba', '\xbb', '\xbc', '\xbd', '\xbe', '\xbf',
                '\xc0', '\xc1', '\xc2', '\xc3', '\xc4', '\xc5', '\xc6', '\xc7',
                '\xc8', '\xc9', '\xca', '\xcb', '\xcc', '\xcd', '\xce', '\xcf',
                '\xd0', '\xd1', '\xd2', '\xd3', '\xd4', '\xd5', '\xd6', '\xd7',
                '\xd8', '\xd9', '\xda', '\xdb', '\xdc', '\xdd', '\xde', '\xdf',
                '\xe0', '\xe1', '\xe2', '\xe3', '\xe4', '\xe5', '\xe6', '\xe7',
                '\xe8', '\xe9', '\xea', '\xeb', '\xec', '\xed', '\xee', '\xef',
                '\xf0', '\xf1', '\xf2', '\xf3', '\xf4', '\xf5', '\xf6', '\xf7',
                '\xf8', '\xf9', '\xfa', '\xfb', '\xfc', '\xfd', '\xfe', '\xff',
        };

        // Declaration for the array of characters to lower-case characters.
        TURBO_DLL constexpr char kToLower[256] = {
                '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
                '\x08', '\x09', '\x0a', '\x0b', '\x0c', '\x0d', '\x0e', '\x0f',
                '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
                '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
                '\x20', '\x21', '\x22', '\x23', '\x24', '\x25', '\x26', '\x27',
                '\x28', '\x29', '\x2a', '\x2b', '\x2c', '\x2d', '\x2e', '\x2f',
                '\x30', '\x31', '\x32', '\x33', '\x34', '\x35', '\x36', '\x37',
                '\x38', '\x39', '\x3a', '\x3b', '\x3c', '\x3d', '\x3e', '\x3f',
                '\x40', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                'x', 'y', 'z', '\x5b', '\x5c', '\x5d', '\x5e', '\x5f',
                '\x60', '\x61', '\x62', '\x63', '\x64', '\x65', '\x66', '\x67',
                '\x68', '\x69', '\x6a', '\x6b', '\x6c', '\x6d', '\x6e', '\x6f',
                '\x70', '\x71', '\x72', '\x73', '\x74', '\x75', '\x76', '\x77',
                '\x78', '\x79', '\x7a', '\x7b', '\x7c', '\x7d', '\x7e', '\x7f',
                '\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87',
                '\x88', '\x89', '\x8a', '\x8b', '\x8c', '\x8d', '\x8e', '\x8f',
                '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97',
                '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9d', '\x9e', '\x9f',
                '\xa0', '\xa1', '\xa2', '\xa3', '\xa4', '\xa5', '\xa6', '\xa7',
                '\xa8', '\xa9', '\xaa', '\xab', '\xac', '\xad', '\xae', '\xaf',
                '\xb0', '\xb1', '\xb2', '\xb3', '\xb4', '\xb5', '\xb6', '\xb7',
                '\xb8', '\xb9', '\xba', '\xbb', '\xbc', '\xbd', '\xbe', '\xbf',
                '\xc0', '\xc1', '\xc2', '\xc3', '\xc4', '\xc5', '\xc6', '\xc7',
                '\xc8', '\xc9', '\xca', '\xcb', '\xcc', '\xcd', '\xce', '\xcf',
                '\xd0', '\xd1', '\xd2', '\xd3', '\xd4', '\xd5', '\xd6', '\xd7',
                '\xd8', '\xd9', '\xda', '\xdb', '\xdc', '\xdd', '\xde', '\xdf',
                '\xe0', '\xe1', '\xe2', '\xe3', '\xe4', '\xe5', '\xe6', '\xe7',
                '\xe8', '\xe9', '\xea', '\xeb', '\xec', '\xed', '\xee', '\xef',
                '\xf0', '\xf1', '\xf2', '\xf3', '\xf4', '\xf5', '\xf6', '\xf7',
                '\xf8', '\xf9', '\xfa', '\xfb', '\xfc', '\xfd', '\xfe', '\xff',
        };

    }  // namespace ascii_internal

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is an alphabetic character.
     */
    constexpr bool ascii_is_alpha(unsigned char c) {
        return (ascii_internal::kPropertyBits[c] & 0x01) != 0;
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is an alphanumeric character.
     */
    constexpr bool ascii_is_alnum(unsigned char c) {
        return (ascii_internal::kPropertyBits[c] & 0x04) != 0;
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a whitespace character (space,
     * tab, vertical tab, formfeed, linefeed, or carriage return).
     */
    constexpr bool ascii_is_space(unsigned char c) {
        return (ascii_internal::kPropertyBits[c] & 0x08) != 0;
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a punctuation character.
     */
    constexpr bool ascii_is_punct(unsigned char c) {
        return (ascii_internal::kPropertyBits[c] & 0x10) != 0;
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a blank character (tab or space).
     */
    constexpr bool ascii_is_blank(unsigned char c) {
        return (ascii_internal::kPropertyBits[c] & 0x20) != 0;
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a control character.
     */
    constexpr bool ascii_is_cntrl(unsigned char c) {
        return (ascii_internal::kPropertyBits[c] & 0x40) != 0;
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character can be represented as a hexadecimal
     * digit character (i.e. {0-9} or {A-F}).
     */
    constexpr bool ascii_is_xdigit(unsigned char c) {
        return (ascii_internal::kPropertyBits[c] & 0x80) != 0;
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character can be represented as a decimal
     * digit character (i.e. {0-9}).
     */
    constexpr bool ascii_is_digit(unsigned char c) { return c >= '0' && c <= '9'; }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is printable, including spaces.
     */
    constexpr bool ascii_is_print(unsigned char c) { return c >= 32 && c < 127; }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character has a graphical representation.
     */
    constexpr bool ascii_is_graph(unsigned char c) { return c > 32 && c < 127; }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is uppercase.
     */
    constexpr bool ascii_is_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is lowercase.
     */
    constexpr bool ascii_is_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is ASCII.
     */
    constexpr bool ascii_is_ascii(unsigned char c) { return c < 128; }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Returns the ASCII character, converting to lower-case if upper-case is
     * passed. Note that characters values > 127 are simply returned.
     */
    constexpr char ascii_to_lower(unsigned char c) {
        return ascii_internal::kToLower[c];
    }


    /**
     * @ingroup turbo_strings_ascii
     * @brief Returns the ASCII character, converting to upper-case if lower-case is
     * passed. Note that characters values > 127 are simply returned.
     */
    constexpr char ascii_to_upper(unsigned char c) {
        return ascii_internal::kToUpper[c];
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a comparison operator.
     */
    constexpr bool ascii_is_operator(char c) noexcept {
        return c == '<' || c == '>' || c == '=';
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a `.`.
     */
    constexpr bool ascii_is_dot(char c) noexcept {
        return c == '.';
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a `|`.
     */
    constexpr bool ascii_is_logical_or(char c) noexcept {
        return c == '|';
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a `&`.
     */
    constexpr bool ascii_is_logical_and(char c) noexcept {
        return c == '&';
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a `^`.
     */
    constexpr bool ascii_is_logical_xor(char c) noexcept {
        return c == '^';
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a `!`.
     */
    constexpr bool ascii_is_logical_not(char c) noexcept {
        return c == '!';
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a hyphen.
     */
    constexpr bool ascii_is_hyphen(char c) noexcept {
        return c == '-';
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Determines whether the given character is a letter.
     */
    constexpr bool ascii_is_letter(char c) noexcept {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }

    /**
     * @ingroup turbo_strings_ascii
     * @brief Converts a character to a digit.
     */
    constexpr std::uint16_t ascii_to_digit(char c) noexcept {
        return static_cast<std::uint16_t>(c - '0');
    }

}  // namespace turbo

#endif  // TURBO_STRINGS_ASCII_H_
