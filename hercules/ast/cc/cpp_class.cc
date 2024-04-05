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


#include <hercules/ast/cc/cpp_class.h>
#include <hercules/ast/cc/cpp_alias_template.h>
#include <hercules/ast/cc/cpp_class_template.h>
#include <hercules/ast/cc/cpp_entity_index.h>
#include <hercules/ast/cc/cpp_entity_kind.h>

namespace hercules::ccast {

    std::unique_ptr<cpp_class> cpp_class::builder::finish(
            const cpp_entity_index &idx, cpp_entity_id id,
            collie::ts::optional<cpp_entity_ref> semantic_parent) {
        class_->set_semantic_parent(std::move(semantic_parent));
        idx.register_definition(std::move(id), collie::ts::ref(*class_));
        return std::move(class_);
    }

    std::unique_ptr<cpp_class> cpp_class::builder::finish_declaration(const cpp_entity_index &idx,
                                                                      cpp_entity_id definition_id) {
        class_->mark_declaration(definition_id);
        idx.register_forward_declaration(std::move(definition_id), collie::ts::ref(*class_));
        return std::move(class_);
    }

    std::unique_ptr<cpp_class> cpp_class::builder::finish(
            collie::ts::optional<cpp_entity_ref> semantic_parent) {
        class_->set_semantic_parent(std::move(semantic_parent));
        return std::move(class_);
    }

    std::unique_ptr<cpp_class> cpp_class::builder::finish_declaration(cpp_entity_id definition_id) {
        class_->mark_declaration(definition_id);
        return std::move(class_);
    }

    const char *to_string(cpp_class_kind kind) noexcept {
        switch (kind) {
            case cpp_class_kind::class_t:
                return "class";
            case cpp_class_kind::struct_t:
                return "struct";
            case cpp_class_kind::union_t:
                return "union";
        }

        return "should not get here";
    }

    const char *to_string(cpp_access_specifier_kind access) noexcept {
        switch (access) {
            case cpp_public:
                return "public";
            case cpp_protected:
                return "protected";
            case cpp_private:
                return "private";
        }

        return "should not get here either";
    }

    cpp_entity_kind cpp_access_specifier::kind() noexcept {
        return cpp_entity_kind::access_specifier_t;
    }

    cpp_entity_kind cpp_access_specifier::do_get_entity_kind() const noexcept {
        return kind();
    }

    cpp_entity_kind cpp_base_class::kind() noexcept {
        return cpp_entity_kind::base_class_t;
    }

    cpp_entity_kind cpp_base_class::do_get_entity_kind() const noexcept {
        return kind();
    }

    cpp_entity_kind cpp_class::kind() noexcept {
        return cpp_entity_kind::class_t;
    }

    cpp_entity_kind cpp_class::do_get_entity_kind() const noexcept {
        return kind();
    }

    namespace {
        cpp_entity_ref get_type_ref(const cpp_type &type) {
            if (type.kind() == cpp_type_kind::user_defined_t) {
                auto &ref = static_cast<const cpp_user_defined_type &>(type).entity();
                return cpp_entity_ref(ref.id()[0u], ref.name());
            } else if (type.kind() == cpp_type_kind::template_instantiation_t) {
                auto &ref = static_cast<const cpp_template_instantiation_type &>(type).primary_template();
                return cpp_entity_ref(ref.id()[0u], ref.name());
            }

            DEBUG_ASSERT(type.kind() == cpp_type_kind::template_parameter_t
                         || type.kind() == cpp_type_kind::decltype_t
                         || type.kind() == cpp_type_kind::decltype_auto_t
                         || type.kind() == cpp_type_kind::unexposed_t,
                         detail::assert_handler{});
            return cpp_entity_ref(cpp_entity_id("<null id>"), "");
        }

        collie::ts::optional_ref<const cpp_entity> get_entity_impl(const cpp_entity_index &index,
                                                                   const cpp_entity_ref &ref) {
            auto result = ref.get(index);
            if (result.empty())
                return nullptr;
            DEBUG_ASSERT(result.size() == 1u, detail::assert_handler{});

            auto entity = result.front();
            if (entity->kind() == cpp_class_template::kind())
                return collie::ts::ref(static_cast<const cpp_class_template &>(*entity).class_());
            else if (entity->kind() == cpp_class_template_specialization::kind())
                return collie::ts::ref(
                        static_cast<const cpp_class_template_specialization &>(*entity).class_());
            else
                return entity;
        }

        collie::ts::optional_ref<const cpp_class> get_class_impl(const cpp_entity_index &index,
                                                                 const cpp_entity_ref &ref) {
            auto entity = get_entity_impl(index, ref);
            if (!entity)
                return nullptr;

            if (entity.value().kind() == cpp_alias_template::kind()) {
                auto &alias = static_cast<const hercules::ccast::cpp_alias_template &>(entity.value());
                return get_class_impl(index, get_type_ref(alias.type_alias().underlying_type()));
            } else if (entity.value().kind() == cpp_type_alias::kind()) {
                auto &alias = static_cast<const hercules::ccast::cpp_type_alias &>(entity.value());
                return get_class_impl(index, get_type_ref(alias.underlying_type()));
            } else {
                DEBUG_ASSERT(entity.value().kind() == cpp_class::kind(), detail::assert_handler{});
                return collie::ts::ref(static_cast<const cpp_class &>(entity.value()));
            }
        }
    } // namespace

    collie::ts::optional_ref<const cpp_class> get_class(const cpp_entity_index &index,
                                                                         const cpp_base_class &base) {
        return get_class_impl(index, get_type_ref(base.type()));
    }

    collie::ts::optional_ref<const cpp_entity> get_class_or_typedef(
            const cpp_entity_index &index, const cpp_base_class &base) {
        return get_entity_impl(index, get_type_ref(base.type()));
    }
}  // namespace hercules::ccast
