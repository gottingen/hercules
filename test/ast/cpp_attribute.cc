// Copyright (C) 2017-2023 Jonathan Müller and cppast contributors
// SPDX-License-Identifier: MIT

#include <hercules/ast/cc/cpp_attribute.h>

#include <hercules/ast/cc/cpp_function.h>
#include <hercules/ast/cc/cpp_variable.h>

#include "test_parser.h"

using namespace hercules::ccast;

TEST_CASE("cpp_attribute")
{
    auto code = R"(
// multiple attributes
[[attribute1]] void [[attribute2]] a();
[[attribute1, attribute2]] void b();

// variadic attributes - not actually supported by clang
//[[variadic...]] void c();

// scoped attributes
[[ns::attribute]] void d();

// argument attributes
[[attribute(arg1, arg2, +(){}, 42, "Hello!")]] void e();

// all of the above
[[ns::attribute(+, -, 0 4), other_attribute]] void f();

// known attributes
[[deprecated]] void g();
[[maybe_unused]] void h();
[[nodiscard]] int i();
[[noreturn]] void j();

// alignas
struct alignas(8) type {};
alignas(type) int var;

// keyword attributes
[[const]] int k();

// multiple attributes but separately
[[a]] [[b]] [[c]] int l();
)";

    auto file = parse({}, "cpp_attribute.cpp", code);

    auto check_attribute
        = [](const cpp_attribute& attr, const char* name, collie::ts::optional<std::string> scope,
             bool variadic, const char* args, cpp_attribute_kind kind) {
              REQUIRE(attr.kind() == kind);
              REQUIRE(attr.name() == name);
              REQUIRE(attr.scope() == scope);
              REQUIRE(attr.is_variadic() == variadic);

              if (attr.arguments())
                  REQUIRE(attr.arguments().value().as_string() == args);
              else
                  REQUIRE(*args == '\0');
          };

    auto count = test_visit<cpp_function>(
        *file,
        [&](const cpp_entity& e) {
            auto& attributes = e.attributes();
            REQUIRE(attributes.size() >= 1u);
            auto& attr = attributes.front();

            if (e.name() == "a" || e.name() == "b")
            {
                REQUIRE(attributes.size() == 2u);
                REQUIRE(has_attribute(e, "attribute1"));
                REQUIRE(has_attribute(e, "attribute2"));
                check_attribute(attr, "attribute1", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::unknown);
                check_attribute(attributes[1u], "attribute2", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::unknown);
            }
            else if (e.name() == "c")
                check_attribute(attr, "variadic", collie::ts::nullopt, true, "",
                                cpp_attribute_kind::unknown);
            else if (e.name() == "d")
            {
                REQUIRE(has_attribute(e, "ns::attribute"));
                check_attribute(attr, "attribute", "ns", false, "", cpp_attribute_kind::unknown);
            }
            else if (e.name() == "e")
                check_attribute(attr, "attribute", collie::ts::nullopt, false,
                                R"(arg1,arg2,+(){},42,"Hello!")", cpp_attribute_kind::unknown);
            else if (e.name() == "f")
            {
                REQUIRE(attributes.size() == 2u);
                check_attribute(attr, "attribute", "ns", false, "+,-,0 4",
                                cpp_attribute_kind::unknown);
                check_attribute(attributes[1u], "other_attribute", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::unknown);
            }
            else if (e.name() == "g")
            {
                REQUIRE(has_attribute(e, cpp_attribute_kind::deprecated));
                check_attribute(attr, "deprecated", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::deprecated);
            }
            else if (e.name() == "h")
                check_attribute(attr, "maybe_unused", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::maybe_unused);
            else if (e.name() == "i")
                check_attribute(attr, "nodiscard", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::nodiscard);
            else if (e.name() == "j")
                check_attribute(attr, "noreturn", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::noreturn);
            else if (e.name() == "k")
                check_attribute(attr, "const", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::unknown);
            else if (e.name() == "l")
            {
                REQUIRE_NOTHROW(attributes.size() == 3);
                check_attribute(attributes[0], "a", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::unknown);
                check_attribute(attributes[1], "b", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::unknown);
                check_attribute(attributes[2], "c", collie::ts::nullopt, false, "",
                                cpp_attribute_kind::unknown);
            }
        },
        false);
    REQUIRE(count == 11);

    count = test_visit<cpp_class>(
        *file,
        [&](const cpp_entity& e) {
            auto& attributes = e.attributes();
            REQUIRE(attributes.size() == 1u);
            auto& attr = attributes.front();
            check_attribute(attr, "alignas", collie::ts::nullopt, false, "8",
                            cpp_attribute_kind::alignas_);
        },
        false);
    REQUIRE(count == 1u);

    count = test_visit<cpp_variable>(
        *file,
        [&](const cpp_entity& e) {
            auto& attributes = e.attributes();
            INFO(e.name());
            REQUIRE(attributes.size() == 1u);
            auto& attr = attributes.front();
            check_attribute(attr, "alignas", collie::ts::nullopt, false, "type",
                            cpp_attribute_kind::alignas_);
        },
        false);
    REQUIRE(count == 1u);
}

TEST_CASE("cpp_attribute matching")
{
    auto code = R"(
// classes
struct [[a]] a {};
class [[b]] b {};

template <typename T>
class [[c]] c {};
template <typename T>
class [[c]] c<T*> {};
template <>
class [[c]] c<int> {};

// enums
enum [[e]] e {};
enum class [[f]] f
{
    a [[a]],
    b [[b]] = 42,
};

// functions
[[g]] void g();
void [[h]] h();
void i [[i]] ();
void j() [[j]];
auto k() -> int [[k]];

struct [[member_functions]] member_functions
{
    void a() [[a]];
    void b() const && [[b]];
    virtual void c() [[c]] final;
    virtual void d() [[d]] = 0;

    [[member_functions]] member_functions();
    member_functions(const member_functions&) [[member_functions]];
};

// variables
[[l]] const int l = 42;
static void* [[m]] m;

void [[function_params]] function_params
([[a]] int a, int [[b]] b, int c [[c]] = 42);

struct [[members]] members
{
    int [[a]] a;
    int [[b]] b : 2;
};

struct [[bases]] bases
: [[a]] public a,
  [[members]] members
{};

// namespace
namespace [[n]] n {}

// type aliases
using o [[o]] = int;

template <typename T>
using p [[p]] = T;

// constructor
struct [[q]] q
{
    [[q]] q();
};

struct [[r]] r
{
    [[r]]
    r();
};

// type defined inline
struct [[inline_type]] inline_type
{
    [[field]] int field;
}
[[s]] s;

int t [[t]];
)";

    auto file = parse({}, "cpp_attribute__matching.cpp", code);

    auto count = 0u;
    auto check = [&](const hercules::ccast::cpp_entity& e) {
        INFO(e.name());
        REQUIRE(e.attributes().size() == 1u);
        REQUIRE(e.attributes().begin()->name() == e.name());
        ++count;
    };

    visit(*file, [&](const hercules::ccast::cpp_entity& e, const hercules::ccast::visitor_info& info) {
        if (info.event != hercules::ccast::visitor_info::container_entity_exit
            && e.kind() != hercules::ccast::cpp_file::kind() && !is_friended(e) && !is_templated(e))
        {
            check(e);
            if (e.kind() == hercules::ccast::cpp_function::kind())
                for (auto& param : static_cast<const hercules::ccast::cpp_function&>(e).parameters())
                    check(param);
            else if (e.kind() == hercules::ccast::cpp_class::kind())
                for (auto& base : static_cast<const hercules::ccast::cpp_class&>(e).bases())
                    check(base);
        }

        return true;
    });
    REQUIRE(count == 44u);
}
