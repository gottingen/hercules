// Copyright (C) 2017-2023 Jonathan Müller and hercules::ccast contributors
// SPDX-License-Identifier: MIT

#ifndef CPPAST_TEST_PARSER_HPP_INCLUDED
#define CPPAST_TEST_PARSER_HPP_INCLUDED

#include <fstream>

#include <collie/testing/doctest.h>

#include <hercules/ast/cc/code_generator.h>
#include <hercules/ast/cc/cpp_class.h>
#include <hercules/ast/cc/cpp_entity_kind.h>
#include <hercules/ast/cc/cpp_expression.h>
#include <hercules/ast/cc/cpp_type.h>
#include <hercules/ast/cc/libclang_parser.h>
#include <hercules/ast/cc/visitor.h>

inline void write_file(const char* name, const char* code)
{
    std::ofstream file(name);
    file << code;
}

inline std::unique_ptr<hercules::ccast::cpp_file> parse_file(const hercules::ccast::cpp_entity_index& idx,
                                                    const char*                     name,
                                                    bool                 fast_preprocessing = false,
                                                             hercules::ccast::cpp_standard standard
                                                    = hercules::ccast::cpp_standard::cpp_latest)
{
    using namespace hercules::ccast;;

    // Creating a config is expensive, so we remember a default one.
    static auto default_config = libclang_compile_config();
    auto        config         = default_config;
    config.set_flags(standard);
    config.fast_preprocessing(fast_preprocessing);

    libclang_parser p(default_logger());

    std::unique_ptr<hercules::ccast::cpp_file> result;
    REQUIRE_NOTHROW(result = p.parse(idx, name, config));
    REQUIRE(!p.error());
    return result;
}

inline std::unique_ptr<hercules::ccast::cpp_file> parse(const hercules::ccast::cpp_entity_index& idx,
                                               const char* name, const char* code,
                                               bool                 fast_preprocessing = false,
                                                        hercules::ccast::cpp_standard standard
                                               = hercules::ccast::cpp_standard::cpp_latest)
{
    write_file(name, code);
    return parse_file(idx, name, fast_preprocessing, standard);
}

class test_generator : public hercules::ccast::code_generator
{
public:
    test_generator(generation_options options) : options_(std::move(options)) {}

    const std::string& str() const noexcept
    {
        return str_;
    }

private:
    generation_options do_get_options(const hercules::ccast::cpp_entity&,
                                      hercules::ccast::cpp_access_specifier_kind) override
    {
        return options_;
    }

    void do_indent() override
    {
        ++indent_;
    }

    void do_unindent() override
    {
        if (indent_)
            --indent_;
    }

    void do_write_token_seq(hercules::ccast::string_view tokens) override
    {
        if (was_newline_)
        {
            str_ += std::string(indent_ * 2u, ' ');
            was_newline_ = false;
        }
        str_ += tokens.c_str();
    }

    void do_write_newline() override
    {
        str_ += "\n";
        was_newline_ = true;
    }

    std::string        str_;
    generation_options options_;
    unsigned           indent_      = 0;
    bool               was_newline_ = false;
};

inline std::string get_code(const hercules::ccast::cpp_entity&                  e,
                            hercules::ccast::code_generator::generation_options options = {})
{
    test_generator generator(options);
    hercules::ccast::generate_code(generator, e);
    auto str = generator.str();
    if (!str.empty() && str.back() == '\n')
        str.pop_back();
    return str;
}

template <typename Func, typename T>
auto visit_callback(bool, Func f, const T& t) -> decltype(f(t) == true)
{
    return f(t);
}

template <typename Func, typename T>
bool visit_callback(int check, Func f, const T& t)
{
    f(t);
    return check == 1;
}

template <typename T, typename Func>
unsigned test_visit(const hercules::ccast::cpp_file& file, Func f, bool check_code = true)
{
    auto count = 0u;
    hercules::ccast::visit(file, [&](const hercules::ccast::cpp_entity& e, hercules::ccast::visitor_info info) {
        if (info.event == hercules::ccast::visitor_info::container_entity_exit)
            return true; // already handled

        if (e.kind() == T::kind())
        {
            auto& obj       = static_cast<const T&>(e);
            auto  check_cur = visit_callback(check_code, f, obj);
            ++count;

            if (check_cur)
            {
                INFO(e.name());
                REQUIRE(e.comment());
                REQUIRE(e.comment().value() == get_code(e));
            }
        }

        return true;
    });

    return count;
}

// number of direct children
template <class Entity>
unsigned count_children(const Entity& cont)
{
    return unsigned(std::distance(cont.begin(), cont.end()));
}

// ignores templated scopes
inline std::string full_name(const hercules::ccast::cpp_entity& e)
{
    if (e.name().empty())
        return "";
    else if (hercules::ccast::is_parameter(e.kind()))
        // parameters don't have a full name
        return e.name();

    std::string scopes;

    for (auto cur = e.parent(); cur; cur = cur.value().parent())
        // prepend each scope, if there is any
        collie::ts::with(cur.value().scope_name(), [&](const hercules::ccast::cpp_scope_name& cur_scope) {
            scopes = cur_scope.name() + "::" + scopes;
        });

    if (e.kind() == hercules::ccast::cpp_entity_kind::class_t)
    {
        auto& c = static_cast<const hercules::ccast::cpp_class&>(e);
        return scopes + c.semantic_scope() + c.name();
    }
    else
        return scopes + e.name();
}

// checks the full name/parent
inline void check_parent(const hercules::ccast::cpp_entity& e, const char* parent_name,
                         const char* full_name)
{
    REQUIRE(e.parent());
    REQUIRE(e.parent().value().name() == parent_name);
    REQUIRE(::full_name(e) == full_name);
}

bool equal_types(const hercules::ccast::cpp_entity_index& idx, const hercules::ccast::cpp_type& parsed,
                 const hercules::ccast::cpp_type& synthesized);

inline bool equal_expressions(const hercules::ccast::cpp_expression& parsed,
                              const hercules::ccast::cpp_expression& synthesized)
{
    using namespace hercules::ccast;;

    if (parsed.kind() != synthesized.kind())
        return false;
    switch (parsed.kind())
    {
    case cpp_expression_kind::unexposed_t:
        return static_cast<const cpp_unexposed_expression&>(parsed).expression().as_string()
               == static_cast<const cpp_unexposed_expression&>(synthesized)
                      .expression()
                      .as_string();

    case cpp_expression_kind::literal_t:
        return static_cast<const cpp_literal_expression&>(parsed).value()
               == static_cast<const cpp_literal_expression&>(synthesized).value();
    }

    return false;
}

template <typename T, class Predicate>
bool equal_ref(const hercules::ccast::cpp_entity_index&                   idx,
               const hercules::ccast::basic_cpp_entity_ref<T, Predicate>& parsed,
               const hercules::ccast::basic_cpp_entity_ref<T, Predicate>& synthesized,
               const char*                                       full_name_override = nullptr)
{
    if (parsed.name() != synthesized.name())
        return false;
    else if (parsed.is_overloaded() != synthesized.is_overloaded())
        return false;
    else if (parsed.is_overloaded())
        return false;

    auto entities = parsed.get(idx);
    if (entities.size() != 1u)
        return false;
    return entities[0u]->name().empty()
           || full_name(*entities[0u]) == (full_name_override ? full_name_override : parsed.name());
}

template <typename T>
void check_template_parameters(
    const T& templ, std::initializer_list<std::pair<hercules::ccast::cpp_entity_kind, const char*>> params)
{
    // no need to check more
    auto cur = params.begin();
    for (auto& param : templ.parameters())
    {
        REQUIRE(cur != params.end());
        REQUIRE(param.kind() == cur->first);
        REQUIRE(param.name() == cur->second);
        ++cur;
    }
    REQUIRE(cur == params.end());
}

#endif // CPPAST_TEST_PARSER_HPP_INCLUDED
