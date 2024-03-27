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


#include <hercules/ast/cc/libclang/debug_helper.h>
#include <cstdio>
#include <mutex>
#include <hercules/ast/cc/libclang/cxtokenizer.h>

using namespace hercules::ccast;

detail::cxstring detail::get_display_name(const CXCursor& cur) noexcept
{
    return cxstring(clang_getCursorDisplayName(cur));
}

detail::cxstring detail::get_cursor_kind_spelling(const CXCursor& cur) noexcept
{
    return cxstring(clang_getCursorKindSpelling(clang_getCursorKind(cur)));
}

detail::cxstring detail::get_type_kind_spelling(const CXType& type) noexcept
{
    return cxstring(clang_getTypeKindSpelling(type.kind));
}

namespace
{
std::mutex mtx;
}

void detail::print_cursor_info(const CXCursor& cur) noexcept
{
    std::lock_guard<std::mutex> lock(mtx);
    std::fprintf(stderr, "[debug] cursor '%s' (%s): %s\n", get_display_name(cur).c_str(),
                 cxstring(clang_getCursorKindSpelling(cur.kind)).c_str(),
                 cxstring(clang_getCursorUSR(cur)).c_str());
}

void detail::print_type_info(const CXType& type) noexcept
{
    std::lock_guard<std::mutex> lock(mtx);
    std::fprintf(stderr, "[debug] type '%s' (%s)\n", cxstring(clang_getTypeSpelling(type)).c_str(),
                 get_type_kind_spelling(type).c_str());
}

void detail::print_tokens(const CXTranslationUnit& tu, const CXFile& file,
                          const CXCursor& cur) noexcept
{
    std::lock_guard<std::mutex> lock(mtx);
    detail::cxtokenizer         tokenizer(tu, file, cur);
    for (auto& token : tokenizer)
        std::fprintf(stderr, "%s ", token.c_str());
    std::fputs("\n", stderr);
}
