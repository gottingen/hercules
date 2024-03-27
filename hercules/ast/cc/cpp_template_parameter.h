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

#include <collie/type_safe/optional.h>
#include <collie/type_safe/variant.h>

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_variable_base.h>
#include <hercules/ast/cc/detail/intrusive_list.h>

namespace hercules::ccast
{
/// Base class for all entities modelling a template parameter of some kind.
class cpp_template_parameter : public cpp_entity
{
public:
    /// \returns Whether or not the parameter is variadic.
    bool is_variadic() const noexcept
    {
        return variadic_;
    }

protected:
    cpp_template_parameter(std::string name, bool variadic)
    : cpp_entity(std::move(name)), variadic_(variadic)
    {}

private:
    bool variadic_;
};

/// The kind of keyword used in a template parameter.
enum class cpp_template_keyword
{
    keyword_class,
    keyword_typename,
    concept_contraint
};

/// \returns The string associated of the keyword.
const char* to_string(cpp_template_keyword kw) noexcept;

/// A [hercules::ccast::cpp_entity]() modelling a C++ template type parameter.
class cpp_template_type_parameter final : public cpp_template_parameter
{
public:
    static cpp_entity_kind kind() noexcept;

    /// \returns A newly created and registered template type parameter.
    /// \notes The `default_type` may be `nullptr` in which case the parameter has no default.
    static std::unique_ptr<cpp_template_type_parameter> build(
        const cpp_entity_index& idx, cpp_entity_id id, std::string name, cpp_template_keyword kw,
        bool variadic, std::unique_ptr<cpp_type> default_type = nullptr,
        collie::ts::optional<cpp_token_string> concept_constraint = collie::ts::nullopt);

    /// \returns A [ts::optional_ref]() to the default type.
    collie::ts::optional_ref<const cpp_type> default_type() const noexcept
    {
        return collie::ts::opt_cref(default_type_.get());
    }

    /// \returns The keyword used in the template parameter.
    cpp_template_keyword keyword() const noexcept
    {
        return keyword_;
    }

    const collie::ts::optional<cpp_token_string>& concept_constraint() const noexcept
    {
        return concept_constraint_;
    }

private:
    cpp_template_type_parameter(std::string name, cpp_template_keyword kw, bool variadic,
                                std::unique_ptr<cpp_type>             default_type,
                                collie::ts::optional<cpp_token_string> concept_constraint)
    : cpp_template_parameter(std::move(name), variadic), default_type_(std::move(default_type)),
      keyword_(kw), concept_constraint_(concept_constraint)
    {}

    cpp_entity_kind do_get_entity_kind() const noexcept override;

    std::unique_ptr<cpp_type>             default_type_;
    cpp_template_keyword                  keyword_;
    collie::ts::optional<cpp_token_string> concept_constraint_;
};

/// \exclude
namespace detail
{
    struct cpp_template_parameter_ref_predicate
    {
        bool operator()(const cpp_entity& e);
    };
} // namespace detail

/// Reference to a [hercules::ccast::cpp_template_type_parameter]().
using cpp_template_type_parameter_ref
    = basic_cpp_entity_ref<cpp_template_type_parameter,
                           detail::cpp_template_parameter_ref_predicate>;

/// A [hercules::ccast::cpp_type]() defined by a [hercules::ccast::cpp_template_type_parameter]().
class cpp_template_parameter_type final : public cpp_type
{
public:
    /// \returns A newly created parameter type.
    static std::unique_ptr<cpp_template_parameter_type> build(
        cpp_template_type_parameter_ref parameter)
    {
        return std::unique_ptr<cpp_template_parameter_type>(
            new cpp_template_parameter_type(std::move(parameter)));
    }

    /// \returns A reference to the [hercules::ccast::cpp_template_type_parameter]() this type refers to.
    const cpp_template_type_parameter_ref& entity() const noexcept
    {
        return parameter_;
    }

private:
    cpp_template_parameter_type(cpp_template_type_parameter_ref parameter)
    : parameter_(std::move(parameter))
    {}

    cpp_type_kind do_get_kind() const noexcept override
    {
        return cpp_type_kind::template_parameter_t;
    }

    cpp_template_type_parameter_ref parameter_;
};

/// A [hercules::ccast::cpp_entity]() modelling a C++ non-type template parameter.
class cpp_non_type_template_parameter final : public cpp_template_parameter,
                                              public cpp_variable_base
{
public:
    static cpp_entity_kind kind() noexcept;

    /// \returns A newly created and registered non type template parameter.
    /// \notes The `default_value` may be `nullptr` in which case the parameter has no default.
    static std::unique_ptr<cpp_non_type_template_parameter> build(
        const cpp_entity_index& idx, cpp_entity_id id, std::string name,
        std::unique_ptr<cpp_type> type, bool is_variadic,
        std::unique_ptr<cpp_expression> default_value = nullptr);

private:
    cpp_non_type_template_parameter(std::string name, std::unique_ptr<cpp_type> type, bool variadic,
                                    std::unique_ptr<cpp_expression> def)
    : cpp_template_parameter(std::move(name), variadic),
      cpp_variable_base(std::move(type), std::move(def))
    {}

    cpp_entity_kind do_get_entity_kind() const noexcept override;
};

/// \exclude
namespace detail
{
    struct cpp_template_ref_predicate
    {
        bool operator()(const cpp_entity& e);
    };
} // namespace detail

/// A reference to a [hercules::ccast::cpp_template]() or a [hercules::ccast::cpp_template_template_parameter]().
using cpp_template_ref = basic_cpp_entity_ref<cpp_entity, detail::cpp_template_ref_predicate>;

/// A [hercules::ccast::cpp_entity]() modelling a C++ template template parameter.
class cpp_template_template_parameter final : public cpp_template_parameter
{
public:
    static cpp_entity_kind kind() noexcept;

    /// Builds a [hercules::ccast::cpp_template_template_parameter]().
    class builder
    {
    public:
        /// \effects Sets the name and whether it is variadic.
        builder(std::string name, bool variadic)
        : parameter_(new cpp_template_template_parameter(std::move(name), variadic))
        {}

        /// \effects Sets the keyword,
        /// default is [cpp_template_keyword::keyword_class]().
        void keyword(cpp_template_keyword kw)
        {
            parameter_->keyword_ = kw;
        }

        /// \effects Adds a parameter to the template.
        void add_parameter(std::unique_ptr<cpp_template_parameter> param)
        {
            parameter_->parameters_.push_back(*parameter_, std::move(param));
        }

        /// \effects Sets the default template.
        void default_template(cpp_template_ref templ)
        {
            parameter_->default_ = std::move(templ);
        }

        /// \effects Registers the parameter in the [hercules::ccast::cpp_entity_index](),
        /// using the given [hercules::ccast::cpp_entity_id]().
        /// \returns The finished parameter.
        std::unique_ptr<cpp_template_template_parameter> finish(const cpp_entity_index& idx,
                                                                cpp_entity_id           id)
        {
            idx.register_definition(std::move(id), collie::ts::ref(*parameter_));
            return std::move(parameter_);
        }

    private:
        std::unique_ptr<cpp_template_template_parameter> parameter_;
    };

    /// \returns An iteratable object containing the template parameters of the template template
    /// parameter.
    detail::iteratable_intrusive_list<cpp_template_parameter> parameters() const noexcept
    {
        return collie::ts::ref(parameters_);
    }

    /// \returns The keyword used in the template parameter.
    cpp_template_keyword keyword() const noexcept
    {
        return keyword_;
    }

    /// \returns A [ts::optional]() that is the default template.
    collie::ts::optional<cpp_template_ref> default_template() const noexcept
    {
        return default_;
    }

private:
    cpp_template_template_parameter(std::string name, bool variadic)
    : cpp_template_parameter(std::move(name), variadic),
      keyword_(cpp_template_keyword::keyword_class)
    {}

    cpp_entity_kind do_get_entity_kind() const noexcept override;

    detail::intrusive_list<cpp_template_parameter> parameters_;
    collie::ts::optional<cpp_template_ref>          default_;
    cpp_template_keyword                           keyword_;
};

/// An argument for a [hercules::ccast::cpp_template_parameter]().
///
/// It is based on a [ts::variant]() of [hercules::ccast::cpp_type]() (for
/// [hercules::ccast::cpp_template_type_parameter]()), [hercules::ccast::cpp_expression]() (for
/// [hercules::ccast::cpp_non_type_template_parameter]()) and [hercules::ccast::cpp_template_ref]() (for
/// [hercules::ccast::cpp_template_template_parameter]().
class cpp_template_argument
{
public:
    /// \effects Initializes it passing a type as argument.
    /// This corresponds to a [hercules::ccast::cpp_template_type_parameter]().
    /// \notes This constructor only participates in overload resolution if `T` is dervied from
    /// [hercules::ccast::cpp_type](). \param 1 \exclude
    template <typename T,
              typename std::enable_if<std::is_base_of<cpp_type, T>::value, int>::type = 0>
    cpp_template_argument(std::unique_ptr<T> type)
    : arg_(std::unique_ptr<cpp_type>(std::move(type)))
    {}

    /// \effects Initializes it passing an expression as argument.
    /// This corresponds to a [hercules::ccast::cpp_non_type_template_parameter]().
    /// \notes This constructor only participates in overload resolution if `T` is dervied from
    /// [hercules::ccast::cpp_expression](). \param 1 \exclude
    template <typename T,
              typename = typename std::enable_if<std::is_base_of<cpp_expression, T>::value>::type>
    cpp_template_argument(std::unique_ptr<T> expr)
    : arg_(std::unique_ptr<cpp_expression>(std::move(expr)))
    {}

    /// \effects Initializes it passing a template as argument.
    /// This corresponds to a [hercules::ccast::cpp_template_template_parameter]().
    cpp_template_argument(cpp_template_ref templ) : arg_(std::move(templ)) {}

    collie::ts::optional_ref<const cpp_type> type() const noexcept
    {
        return arg_.optional_value(collie::ts::variant_type<std::unique_ptr<cpp_type>>{})
            .map([](const std::unique_ptr<cpp_type>& type) { return collie::ts::ref(*type); });
    }

    collie::ts::optional_ref<const cpp_expression> expression() const noexcept
    {
        return arg_.optional_value(collie::ts::variant_type<std::unique_ptr<cpp_expression>>{})
            .map([](const std::unique_ptr<cpp_expression>& expr) { return collie::ts::ref(*expr); });
    }

    collie::ts::optional_ref<const cpp_template_ref> template_ref() const noexcept
    {
        return arg_.optional_value(collie::ts::variant_type<cpp_template_ref>{});
    }

private:
    collie::ts::variant<std::unique_ptr<cpp_type>, std::unique_ptr<cpp_expression>, cpp_template_ref>
        arg_;
};
} // namespace hercules::ccast

