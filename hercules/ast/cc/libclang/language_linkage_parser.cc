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


#include <hercules/ast/cc/libclang/parse_functions.h>
#include <clang-c/Index.h>
#include <hercules/ast/cc/cpp_language_linkage.h>
#include <hercules/ast/cc/libclang/libclang_visitor.h>

namespace hercules::ccast {

    std::unique_ptr<cpp_entity> detail::try_parse_cpp_language_linkage(const parse_context &context,
                                                                       const CXCursor &cur) {
        DEBUG_ASSERT(cur.kind == CXCursor_UnexposedDecl,
                     detail::assert_handler{}); // not exposed currently

        detail::cxtokenizer tokenizer(context.tu, context.file, cur);
        detail::cxtoken_stream stream(tokenizer, cur);

        // extern <name> ...
        if (!detail::skip_if(stream, "extern"))
            return nullptr;
        // unexposed variable starting with extern - must be a language linkage
        // (function, variables are not unexposed)
        auto &name = stream.get().value();

        auto builder = cpp_language_linkage::builder(name.c_str());
        context.comments.match(builder.get(), cur);
        detail::visit_children(cur, [&](const CXCursor &child) {
            auto entity = parse_entity(context, &builder.get(), child);
            if (entity)
                builder.add_child(std::move(entity));
        });

        return builder.finish();
    }
}  // namespace hercules::ccast
