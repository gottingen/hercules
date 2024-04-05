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

//# Note: these would be included transitively as with any normal C++ project but
//# they're listed explicitly here because this file is used as the source for generate_single_header.py.

#include <collie/toml/impl/preprocessor.h>

TOML_PUSH_WARNINGS;
TOML_DISABLE_SPAM_WARNINGS;
TOML_DISABLE_SWITCH_WARNINGS;
TOML_DISABLE_SUGGEST_ATTR_WARNINGS;

// misc warning false-positives
#if TOML_MSVC
#pragma warning(disable : 5031) // #pragma warning(pop): likely mismatch
#if TOML_SHARED_LIB
#pragma warning(disable : 4251) // dll exports for std lib types
#endif
#elif TOML_CLANG
#pragma clang diagnostic ignored "-Wheader-hygiene"
#if TOML_CLANG >= 12
#pragma clang diagnostic ignored "-Wc++20-extensions"
#endif
#if (TOML_CLANG == 13) && !defined(__APPLE__)
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif
#endif

#include <collie/toml/impl/std_new.h>
#include <collie/toml/impl/std_string.h>
#include <collie/toml/impl/std_optional.h>
#include <collie/toml/impl/forward_declarations.h>
#include <collie/toml/impl/print_to_stream.h>
#include <collie/toml/impl/source_region.h>
#include <collie/toml/impl/date_time.h>
#include <collie/toml/impl/at_path.h>
#include <collie/toml/impl/path.h>
#include <collie/toml/impl/node.h>
#include <collie/toml/impl/node_view.h>
#include <collie/toml/impl/value.h>
#include <collie/toml/impl/make_node.h>
#include <collie/toml/impl/array.h>
#include <collie/toml/impl/key.h>
#include <collie/toml/impl/table.h>
#include <collie/toml/impl/unicode_autogenerated.h>
#include <collie/toml/impl/unicode.h>
#include <collie/toml/impl/parse_error.h>
#include <collie/toml/impl/parse_result.h>
#include <collie/toml/impl/parser.h>
#include <collie/toml/impl/formatter.h>
#include <collie/toml/impl/toml_formatter.h>
#include <collie/toml/impl/json_formatter.h>
#include <collie/toml/impl/yaml_formatter.h>

#if TOML_IMPLEMENTATION

#include <collie/toml/impl/std_string.inl>
#include <collie/toml/impl/print_to_stream.inl>
#include <collie/toml/impl/node.inl>
#include <collie/toml/impl/at_path.inl>
#include <collie/toml/impl/path.inl>
#include <collie/toml/impl/array.inl>
#include <collie/toml/impl/table.inl>
#include <collie/toml/impl/unicode.inl>
#include <collie/toml/impl/parser.inl>
#include <collie/toml/impl/formatter.inl>
#include <collie/toml/impl/toml_formatter.inl>
#include <collie/toml/impl/json_formatter.inl>
#include <collie/toml/impl/yaml_formatter.inl>

#endif // TOML_IMPLEMENTATION

TOML_POP_WARNINGS;

// macro hygiene
#if TOML_UNDEF_MACROS
#undef TOML_ABI_NAMESPACE_BOOL
#undef TOML_ABI_NAMESPACE_END
#undef TOML_ABI_NAMESPACE_START
#undef TOML_ABI_NAMESPACES
#undef TOML_ABSTRACT_INTERFACE
#undef TOML_ALWAYS_INLINE
#undef TOML_ANON_NAMESPACE
#undef TOML_ANON_NAMESPACE_END
#undef TOML_ANON_NAMESPACE_START
#undef TOML_ARM
#undef TOML_ASSERT
#undef TOML_ASSERT_ASSUME
#undef TOML_ASSUME
#undef TOML_ASYMMETRICAL_EQUALITY_OPS
#undef TOML_ATTR
#undef TOML_CLANG
#undef TOML_CLOSED_ENUM
#undef TOML_CLOSED_FLAGS_ENUM
#undef TOML_COMPILER_HAS_EXCEPTIONS
#undef TOML_COMPILER_HAS_RTTI
#undef TOML_CONST
#undef TOML_CONST_GETTER
#undef TOML_CONST_INLINE_GETTER
#undef TOML_CONSTRAINED_TEMPLATE
#undef TOML_CPP
#undef TOML_DECLSPEC
#undef TOML_DELETE_DEFAULTS
#undef TOML_DISABLE_ARITHMETIC_WARNINGS
#undef TOML_DISABLE_CODE_ANALYSIS_WARNINGS
#undef TOML_DISABLE_SPAM_WARNINGS
#undef TOML_DISABLE_SPAM_WARNINGS_CLANG_10
#undef TOML_DISABLE_SPAM_WARNINGS_CLANG_11
#undef TOML_DISABLE_SUGGEST_ATTR_WARNINGS
#undef TOML_DISABLE_SWITCH_WARNINGS
#undef TOML_DISABLE_WARNINGS
#undef TOML_DOXYGEN
#undef TOML_EMPTY_BASES
#undef TOML_ENABLE_IF
#undef TOML_ENABLE_WARNINGS
#undef TOML_EVAL_BOOL_0
#undef TOML_EVAL_BOOL_1
#undef TOML_EXTERNAL_LINKAGE
#undef TOML_FLAGS_ENUM
#undef TOML_FLOAT_CHARCONV
#undef TOML_FLOAT128
#undef TOML_FLOAT16
#undef TOML_FP16
#undef TOML_GCC
#undef TOML_HAS_ATTR
#undef TOML_HAS_BUILTIN
#undef TOML_HAS_CHAR8
#undef TOML_HAS_CPP_ATTR
#undef TOML_HAS_CUSTOM_OPTIONAL_TYPE
#undef TOML_HAS_FEATURE
#undef TOML_HAS_INCLUDE
#undef TOML_HAS_SSE2
#undef TOML_HAS_SSE4_1
#undef TOML_HIDDEN_CONSTRAINT
#undef TOML_ICC
#undef TOML_ICC_CL
#undef TOML_IMPL_NAMESPACE_END
#undef TOML_IMPL_NAMESPACE_START
#undef TOML_IMPLEMENTATION
#undef TOML_INCLUDE_WINDOWS_H
#undef TOML_INT_CHARCONV
#undef TOML_INT128
#undef TOML_INTELLISENSE
#undef TOML_INTERNAL_LINKAGE
#undef TOML_LANG_AT_LEAST
#undef TOML_LANG_EFFECTIVE_VERSION
#undef TOML_LANG_HIGHER_THAN
#undef TOML_LANG_UNRELEASED
#undef TOML_LAUNDER
#undef TOML_LIFETIME_HOOKS
#undef TOML_LIKELY
#undef TOML_LIKELY_CASE
#undef TOML_MAKE_FLAGS
#undef TOML_MAKE_FLAGS_
#undef TOML_MAKE_FLAGS_1
#undef TOML_MAKE_FLAGS_2
#undef TOML_MAKE_STRING
#undef TOML_MAKE_STRING_1
#undef TOML_MAKE_VERSION
#undef TOML_MSVC
#undef TOML_NAMESPACE
#undef TOML_NEVER_INLINE
#undef TOML_NODISCARD
#undef TOML_NODISCARD_CTOR
#undef TOML_OPEN_ENUM
#undef TOML_OPEN_FLAGS_ENUM
#undef TOML_PARSER_TYPENAME
#undef TOML_POP_WARNINGS
#undef TOML_PRAGMA_CLANG
#undef TOML_PRAGMA_CLANG_GE_10
#undef TOML_PRAGMA_CLANG_GE_11
#undef TOML_PRAGMA_CLANG_GE_9
#undef TOML_PRAGMA_GCC
#undef TOML_PRAGMA_ICC
#undef TOML_PRAGMA_MSVC
#undef TOML_PURE
#undef TOML_PURE_GETTER
#undef TOML_PURE_INLINE_GETTER
#undef TOML_PUSH_WARNINGS
#undef TOML_REQUIRES
#undef TOML_SA_LIST_BEG
#undef TOML_SA_LIST_END
#undef TOML_SA_LIST_NEW
#undef TOML_SA_LIST_NXT
#undef TOML_SA_LIST_SEP
#undef TOML_SA_NATIVE_VALUE_TYPE_LIST
#undef TOML_SA_NEWLINE
#undef TOML_SA_NODE_TYPE_LIST
#undef TOML_SA_UNWRAPPED_NODE_TYPE_LIST
#undef TOML_SA_VALUE_EXACT_FUNC_MESSAGE
#undef TOML_SA_VALUE_FUNC_MESSAGE
#undef TOML_SA_VALUE_MESSAGE_CONST_CHAR8
#undef TOML_SA_VALUE_MESSAGE_U8STRING_VIEW
#undef TOML_SA_VALUE_MESSAGE_WSTRING
#undef TOML_SIMPLE_STATIC_ASSERT_MESSAGES
#undef TOML_TRIVIAL_ABI
#undef TOML_UINT128
#undef TOML_UNLIKELY
#undef TOML_UNLIKELY_CASE
#undef TOML_UNREACHABLE
#undef TOML_UNUSED
#undef TOML_WINDOWS
#endif
