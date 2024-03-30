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


#include <hercules/ast/cc/cpp_friend.h>
#include <hercules/ast/cc/cpp_template.h>
#include <hercules/ast/cc/cpp_template_parameter.h>
#include <hercules/ast/cc/libclang/libclang_visitor.h>
#include <hercules/ast/cc/libclang/parse_functions.h>

namespace hercules::ccast {

    std::unique_ptr<cpp_entity> detail::parse_cpp_friend(const detail::parse_context &context,
                                                         const CXCursor &cur) {
        DEBUG_ASSERT(clang_getCursorKind(cur) == CXCursor_FriendDecl, detail::assert_handler{});

        std::string comment;
        std::unique_ptr<cpp_entity> entity;
        std::unique_ptr<cpp_type> type;
        std::string namespace_str;
        collie::ts::optional<cpp_template_instantiation_type::builder> inst_builder;
        detail::visit_children(cur, [&](const CXCursor &child) {
            auto kind = clang_getCursorKind(child);
            if (kind == CXCursor_TypeRef) {
                auto referenced = clang_getCursorReferenced(child);
                if (inst_builder) {
                    namespace_str.clear();
                    inst_builder.value().add_argument(
                            detail::parse_type(context, referenced, clang_getCursorType(referenced)));
                } else if (clang_getCursorKind(referenced) == CXCursor_TemplateTypeParameter)
                    // parse template parameter type
                    type = cpp_template_parameter_type::build(
                            cpp_template_type_parameter_ref(detail::get_entity_id(referenced),
                                                            detail::get_cursor_name(child).c_str()));
                else if (!namespace_str.empty()) {
                    // parse as user defined type
                    // we can't use the other branch here,
                    // as then the class name would be wrong
                    auto name = detail::get_cursor_name(referenced);
                    type = cpp_user_defined_type::build(
                            cpp_type_ref(detail::get_entity_id(referenced),
                                         namespace_str + "::" + name.c_str()));
                } else {
                    // for some reason libclang gives a type ref here
                    // we actually need a class decl cursor, so parse the referenced one
                    // this might be a definition, so give friend information to the parser
                    entity = parse_entity(context, nullptr, referenced, cur);
                    comment = collie::ts::copy(entity->comment()).value_or("");
                }
            } else if (kind == CXCursor_NamespaceRef)
                namespace_str += detail::get_cursor_name(child).c_str();
            else if (kind == CXCursor_TemplateRef) {
                if (!namespace_str.empty())
                    namespace_str += "::";
                auto templ = cpp_template_ref(detail::get_entity_id(clang_getCursorReferenced(child)),
                                              namespace_str + detail::get_cursor_name(child).c_str());
                namespace_str.clear();
                if (!inst_builder)
                    inst_builder = cpp_template_instantiation_type::builder(std::move(templ));
                else
                    inst_builder.value().add_argument(std::move(templ));
            } else if (clang_isDeclaration(kind)) {
                entity = parse_entity(context, nullptr, child, cur);
                if (entity) {
                    // steal comment
                    comment = collie::ts::copy(entity->comment()).value_or("");
                    entity->set_comment(collie::ts::nullopt);
                }
            } else if (inst_builder && clang_isExpression(kind)) {
                namespace_str.clear();
                inst_builder.value().add_argument(detail::parse_expression(context, child));
            }
        });

        std::unique_ptr<cpp_entity> result;
        if (entity)
            result = cpp_friend::build(std::move(entity));
        else if (type)
            result = cpp_friend::build(std::move(type));
        else if (inst_builder)
            result = cpp_friend::build(inst_builder.value().finish());
        else
            DEBUG_UNREACHABLE(detail::parse_error_handler{}, cur,
                              "unknown child entity of friend declaration");
        if (!comment.empty())
            // set comment of entity...
            result->set_comment(std::move(comment));
        // ... but override if this finds a different comment
        // due to clang_getCursorReferenced(), this may happen
        context.comments.match(*result, cur);
        return result;
    }
}  // namespace hercules::ccast
