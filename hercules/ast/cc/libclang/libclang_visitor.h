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

#include <clang-c/Index.h>

#include "raii_wrapper.h"

namespace hercules::ccast
{
namespace detail
{
    // visits direct children of an entity
    template <typename Func>
    void visit_children(CXCursor parent, Func f, bool recurse = false)
    {
        auto continue_lambda = [](CXCursor cur, CXCursor, CXClientData data) {
            auto& actual_cb = *static_cast<Func*>(data);
            actual_cb(cur);
            return CXChildVisit_Continue;
        };
        auto recurse_lambda = [](CXCursor cur, CXCursor, CXClientData data) {
            auto& actual_cb = *static_cast<Func*>(data);
            actual_cb(cur);
            return CXChildVisit_Recurse;
        };

        if (recurse)
            clang_visitChildren(parent, recurse_lambda, &f);
        else
            clang_visitChildren(parent, continue_lambda, &f);
    }

    // visits a translation unit
    // notes: only visits if directly defined in file, not included
    template <typename Func>
    void visit_tu(const cxtranslation_unit& tu, const char* path, Func f)
    {
        auto in_tu = [&](const CXCursor& cur) {
            auto location = clang_getCursorLocation(cur);

            CXString cx_file_name;
            clang_getPresumedLocation(location, &cx_file_name, nullptr, nullptr);
            cxstring file_name(cx_file_name);

            return file_name == path;
        };

        visit_children(clang_getTranslationUnitCursor(tu.get()), [&](const CXCursor& cur) {
            if (in_tu(cur))
                f(cur);
        });
    }
} // namespace detail
} // namespace hercules::ccast
