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

#include <hercules/ast/cc/cpp_token.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <collie/type_safe/optional.h>

#include <hercules/ast/cc/detail/assert.h>

using namespace hercules::ccast;

void cpp_token_string::builder::unmunch()
{
    DEBUG_ASSERT(!tokens_.empty() && tokens_.back().spelling == ">>", detail::assert_handler{});
    tokens_.back().spelling = ">";
}

namespace
{
template <std::size_t N>
bool starts_with(const char* ptr, const char (&str)[N])
{
    return std::strncmp(ptr, str, N - 1u) == 0;
}

bool starts_with(const char* ptr, const std::string& str)
{
    return std::strncmp(ptr, str.c_str(), str.size()) == 0;
}

template <std::size_t N>
bool bump_if(const char*& ptr, const char (&str)[N])
{
    if (starts_with(ptr, str))
    {
        ptr += N - 1;
        return true;
    }
    else
        return false;
}

bool bump_if(const char*& ptr, const std::string& str)
{
    if (starts_with(ptr, str))
    {
        ptr += str.size();
        return true;
    }
    else
        return false;
}

bool is_identifier_nondigit(char c)
{
    // assume ASCII
    if (c >= 'a' && c <= 'z')
        return true;
    else if (c >= 'A' && c <= 'Z')
        return true;
    else if (c == '_')
        return true;
    else
        // technically \uXXX is allowed as well, but I haven't seen that used ever
        return false;
}

bool is_digit(char c)
{
    return c >= '0' && c <= '9';
}

bool is_hexadecimal_digit(char c)
{
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

collie::ts::optional<std::string> bump_identifier(const char*& ptr)
{
    if (is_identifier_nondigit(*ptr))
    {
        std::string result;
        result += *ptr++;

        while (is_identifier_nondigit(*ptr) || is_digit(*ptr))
            result += *ptr++;

        return result;
    }
    else
        return collie::ts::nullopt;
}

collie::ts::optional<cpp_token> identifier_token(const char*& ptr)
{
    auto identifier = bump_identifier(ptr);
    if (!identifier)
        return collie::ts::nullopt;

    static constexpr const char* keywords[] = {"alignas",
                                               "alignof",
                                               "asm",
                                               "auto",
                                               "bool",
                                               "break",
                                               "case",
                                               "catch",
                                               "char",
                                               "char16_t",
                                               "char32_t",
                                               "class",
                                               "const",
                                               "constexpr",
                                               "const_cast",
                                               "continue",
                                               "decltype",
                                               "default",
                                               "delete",
                                               "do",
                                               "double",
                                               "dynamic_cast",
                                               "else",
                                               "enum",
                                               "explicit",
                                               "export",
                                               "extern",
                                               "false",
                                               "float",
                                               "for",
                                               "friend",
                                               "goto",
                                               "if",
                                               "inline",
                                               "int",
                                               "long",
                                               "mutable",
                                               "namespace",
                                               "new",
                                               "noexcept",
                                               "nullptr",
                                               "operator",
                                               "private",
                                               "protected",
                                               "public",
                                               "register",
                                               "reinterpret_cast",
                                               "return",
                                               "short",
                                               "signed",
                                               "sizeof",
                                               "static",
                                               "static_assert",
                                               "static_cast",
                                               "struct",
                                               "switch",
                                               "template",
                                               "this",
                                               "thread_local",
                                               "throw",
                                               "true",
                                               "try",
                                               "typedef",
                                               "typeid",
                                               "typename",
                                               "union",
                                               "unsigned",
                                               "using",
                                               "virtual",
                                               "void",
                                               "volatile",
                                               "wchar_t",
                                               "while"};
    auto find_keyword = std::find(std::begin(keywords), std::end(keywords), identifier.value());
    if (find_keyword != std::end(keywords))
        return cpp_token(cpp_token_kind::keyword, identifier.value());
    else if (identifier == "and")
        return cpp_token(cpp_token_kind::punctuation, "&&");
    else if (identifier == "and_eq")
        return cpp_token(cpp_token_kind::punctuation, "&=");
    else if (identifier == "bitand")
        return cpp_token(cpp_token_kind::punctuation, "&");
    else if (identifier == "bitor")
        return cpp_token(cpp_token_kind::punctuation, "|");
    else if (identifier == "compl")
        return cpp_token(cpp_token_kind::punctuation, "~");
    else if (identifier == "not")
        return cpp_token(cpp_token_kind::punctuation, "!");
    else if (identifier == "not_eq")
        return cpp_token(cpp_token_kind::punctuation, "!=");
    else if (identifier == "or")
        return cpp_token(cpp_token_kind::punctuation, "||");
    else if (identifier == "or_eq")
        return cpp_token(cpp_token_kind::punctuation, "|=");
    else if (identifier == "xor")
        return cpp_token(cpp_token_kind::punctuation, "^");
    else if (identifier == "xor_eq")
        return cpp_token(cpp_token_kind::punctuation, "^=");
    else
        return cpp_token(cpp_token_kind::identifier, identifier.value());
}

void append_udl_suffix(std::string& literal, const char*& ptr)
{
    if (auto id = identifier_token(ptr))
        literal += id.value().spelling;
}

template <typename DigitPredicate>
std::string parse_digit_sequence(const char*& ptr, DigitPredicate is_digit)
{
    std::string result;
    for (; is_digit(*ptr) || *ptr == '\''; ++ptr)
        if (*ptr != '\'')
            result += *ptr;
    DEBUG_ASSERT(result.empty() || result.back() != '\'', detail::assert_handler{});
    return result;
}

void append_integer_suffix(std::string& literal, const char*& ptr)
{
    auto append_unsigned_suffix = [](std::string& literal, const char*& ptr) {
        if (*ptr == 'u' || *ptr == 'U')
        {
            literal += *ptr++;
            return true;
        }
        else
            return false;
    };
    auto append_long_suffix = [](std::string& literal, const char*& ptr) {
        if (starts_with(ptr, "ll") || starts_with(ptr, "LL"))
        {
            literal += *ptr++;
            literal += *ptr++;
            return true;
        }
        else if (*ptr == 'l' || *ptr == 'L')
        {
            literal += *ptr++;
            return true;
        }
        else
            return false;
    };

    if (append_unsigned_suffix(literal, ptr))
        append_long_suffix(literal, ptr);
    else if (append_long_suffix(literal, ptr))
        append_unsigned_suffix(literal, ptr);
    else
        append_udl_suffix(literal, ptr);
}

void append_floating_point_suffix(std::string& literal, const char*& ptr)
{
    if (*ptr == 'f' || *ptr == 'F')
        literal += *ptr++;
    else if (*ptr == 'l' || *ptr == 'L')
        literal += *ptr++;
    else
        append_udl_suffix(literal, ptr);
}

collie::ts::optional<std::string> parse_floating_point_exponent(const char*& ptr)
{
    if (*ptr == 'e' || *ptr == 'E' || *ptr == 'p' || *ptr == 'P')
    {
        std::string result;
        result += *ptr++;
        if (*ptr == '+' || *ptr == '-')
            result += *ptr++;

        result += parse_digit_sequence(ptr, &is_digit);
        return result;
    }
    else
        return collie::ts::nullopt;
}

collie::ts::optional<cpp_token> numeric_literal_token(const char*& ptr)
{
    if (starts_with(ptr, "0b") || starts_with(ptr, "0B")) // binary integer literal
    {
        std::string result;
        result += *ptr++;
        result += *ptr++;
        result += parse_digit_sequence(ptr, [](char c) { return c == '0' || c == '1'; });
        append_integer_suffix(result, ptr);
        return cpp_token(cpp_token_kind::int_literal, result);
    }
    else if (starts_with(ptr, "0x") || starts_with(ptr, "0X")) // hexadecimal literal
    {
        std::string result;
        result += *ptr++;
        result += *ptr++;
        result += parse_digit_sequence(ptr, &is_hexadecimal_digit);

        auto is_float = false;
        if (*ptr == '.')
        {
            // floating point hexadecimal
            is_float = true;
            result += *ptr++;
            result += parse_digit_sequence(ptr, &is_hexadecimal_digit);
        }

        if (auto exp = parse_floating_point_exponent(ptr))
        {
            is_float = true;
            // floating point exponent
            result += exp.value();
        }

        if (is_float)
            append_floating_point_suffix(result, ptr);
        else
            append_integer_suffix(result, ptr);

        return cpp_token(is_float ? cpp_token_kind::float_literal : cpp_token_kind::int_literal,
                         result);
    }
    else if (is_digit(*ptr)) // octal and decimal literals
    {
        std::string result;
        result += parse_digit_sequence(ptr, &is_digit);

        auto is_float = false;
        if (*ptr == '.')
        {
            // floating point decimal
            is_float = true;
            result += *ptr++;
            result += parse_digit_sequence(ptr, &is_hexadecimal_digit);
        }

        if (auto exp = parse_floating_point_exponent(ptr))
        {
            // floating point exponent
            is_float = true;
            result += exp.value();
        }

        if (is_float)
            append_floating_point_suffix(result, ptr);
        else
            append_integer_suffix(result, ptr);

        return cpp_token(is_float ? cpp_token_kind::float_literal : cpp_token_kind::int_literal,
                         result);
    }
    else if (*ptr == '.' && is_digit(ptr[1]))
    {
        std::string result;

        // floating point fraction
        result += *ptr++;
        result += parse_digit_sequence(ptr, &is_digit);

        if (auto exp = parse_floating_point_exponent(ptr))
            result += exp.value();

        append_floating_point_suffix(result, ptr);
        return cpp_token(cpp_token_kind::float_literal, result);
    }
    else
        return collie::ts::nullopt;
}

collie::ts::optional<std::string> parse_encoding_prefix(const char*& ptr)
{
    if (bump_if(ptr, "u8"))
        return "u8";
    else if (bump_if(ptr, "u"))
        return "u";
    else if (bump_if(ptr, "U"))
        return "U";
    else if (bump_if(ptr, "L"))
        return "L";
    else
        return collie::ts::nullopt;
}

collie::ts::optional<cpp_token> character_literal(const char*& ptr)
{
    auto save   = ptr;
    auto prefix = parse_encoding_prefix(ptr);
    if (*ptr != '\'')
    {
        ptr = save;
        return collie::ts::nullopt;
    }
    else
    {
        auto result = prefix.value_or("");
        result += *ptr++;

        while (*ptr != '\'')
        {
            DEBUG_ASSERT(*ptr, detail::assert_handler{});

            if (*ptr == '\\')
                result += *ptr++;
            result += *ptr++;
        }
        result += *ptr++;

        append_udl_suffix(result, ptr);
        return cpp_token(cpp_token_kind::char_literal, result);
    }
}

collie::ts::optional<cpp_token> string_literal(const char*& ptr)
{
    auto save   = ptr;
    auto prefix = parse_encoding_prefix(ptr);
    if (starts_with(ptr, "R\""))
    {
        // raw string literal
        auto result = prefix.value_or("");
        result += *ptr++;
        result += *ptr++;

        std::string terminator;
        terminator += ")";
        while (*ptr != '(')
        {
            result += *ptr;
            terminator += *ptr++;
        }
        result += *ptr++;
        terminator += '"';

        while (!bump_if(ptr, terminator))
        {
            DEBUG_ASSERT(ptr, detail::assert_handler{});
            result += *ptr++;
        }
        result += terminator;

        append_udl_suffix(result, ptr);
        return cpp_token(cpp_token_kind::string_literal, result);
    }
    else if (starts_with(ptr, "\""))
    {
        // regular string literal
        auto result = prefix.value_or("");
        result += *ptr++;

        while (*ptr != '"')
        {
            DEBUG_ASSERT(*ptr, detail::assert_handler{});

            if (*ptr == '\\')
                result += *ptr++;
            result += *ptr++;
        }
        result += *ptr++;

        append_udl_suffix(result, ptr);
        return cpp_token(cpp_token_kind::string_literal, result);
    }
    else
    {
        ptr = save;
        return collie::ts::nullopt;
    }
}

collie::ts::optional<cpp_token> digraph_token(const char*& ptr)
{
    if (bump_if(ptr, "<%"))
        return cpp_token(cpp_token_kind::punctuation, "{");
    else if (bump_if(ptr, "%>"))
        return cpp_token(cpp_token_kind::punctuation, "}");
    else if (starts_with(ptr, "<::") && ptr[3] != ':' && ptr[3] != '>')
        // don't detect digraph in std::vector<::std::string>
        return collie::ts::nullopt;
    else if (bump_if(ptr, "<:"))
        return cpp_token(cpp_token_kind::punctuation, "[");
    else if (bump_if(ptr, ":>"))
        return cpp_token(cpp_token_kind::punctuation, "]");
    else if (bump_if(ptr, "%:%:"))
        return cpp_token(cpp_token_kind::punctuation, "##");
    else if (bump_if(ptr, "%:"))
        return cpp_token(cpp_token_kind::punctuation, "#");
    else
        return collie::ts::nullopt;
}

collie::ts::optional<cpp_token> punctuation_token(const char*& ptr)
{
    static constexpr const char* punctuations[] = {
        // tokens staring with #
        "##",
        "#",
        // tokens starting with .
        "...",
        ".*",
        ".",
        // tokens starting with :
        "::",
        ":",
        // tokens starting with +
        "+=",
        "++",
        "+",
        // tokens starting with -
        "->*",
        "->",
        "--",
        "-=",
        "-",
        // tokens starting with *
        "*=",
        "*",
        // tokens starting with /
        "/=",
        "/",
        // tokens starting with %
        "%=",
        "%",
        // tokens starting with ^
        "^=",
        "^",
        // tokens starting with &
        "&=",
        "&&",
        "&",
        // tokens starting with |
        "|=",
        "||",
        "|",
        // tokens starting with <
        "<<=",
        "<<",
        "<=",
        "<",
        // tokens starting with >
        ">>=",
        ">>",
        ">=",
        ">",
        // tokens starting with !
        "!=",
        "!",
        // tokens starting with =
        "==",
        "=",
        // single tokens
        "~",
        ";",
        "?",
        ",",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
    };

    for (auto punct : punctuations)
        if (bump_if(ptr, punct))
            return cpp_token(cpp_token_kind::punctuation, punct);

    return collie::ts::nullopt;
}
} // namespace

cpp_token_string cpp_token_string::tokenize(std::string str)
{
    cpp_token_string::builder builder;

    auto ptr = str.c_str();
    while (*ptr)
    {
        if (auto num = numeric_literal_token(ptr))
            builder.add_token(num.value());
        else if (auto char_lit = character_literal(ptr))
            builder.add_token(char_lit.value());
        else if (auto str_lit = string_literal(ptr))
            builder.add_token(str_lit.value());
        else if (auto digraphs = digraph_token(ptr))
            builder.add_token(digraphs.value());
        else if (auto punct = punctuation_token(ptr))
            builder.add_token(punct.value());
        else if (auto id = identifier_token(ptr))
            builder.add_token(id.value());
        else if (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')
            ++ptr;
        else
            DEBUG_UNREACHABLE(detail::assert_handler{});
    }

    return builder.finish();
}

namespace
{
bool is_identifier(char c)
{
    return std::isalnum(c) || c == '_';
}
} // namespace

std::string cpp_token_string::as_string() const
{
    std::string result;
    for (auto& token : tokens_)
    {
        DEBUG_ASSERT(!token.spelling.empty(), detail::assert_handler{});
        if (!result.empty() && is_identifier(result.back()) && is_identifier(token.spelling[0u]))
            result += ' ';
        result += token.spelling;
    }
    return result;
}

bool hercules::ccast::operator==(const cpp_token_string& lhs, const cpp_token_string& rhs)
{
    if (lhs.tokens_.size() != rhs.tokens_.size())
        return false;
    return std::equal(lhs.tokens_.begin(), lhs.tokens_.end(), rhs.tokens_.begin());
}
