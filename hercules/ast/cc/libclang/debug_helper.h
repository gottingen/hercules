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

#include <string>

#include <hercules/ast/cc/libclang/raii_wrapper.h>

namespace hercules::ccast {
    namespace detail {
        cxstring get_display_name(const CXCursor &cur) noexcept;

        cxstring get_cursor_kind_spelling(const CXCursor &cur) noexcept;

        cxstring get_type_kind_spelling(const CXType &type) noexcept;

        void print_cursor_info(const CXCursor &cur) noexcept;

        void print_type_info(const CXType &type) noexcept;

        void print_tokens(const CXTranslationUnit &tu, const CXFile &file,
                          const CXCursor &cur) noexcept;
    } // namespace detail
} // namespace hercules::ccast
