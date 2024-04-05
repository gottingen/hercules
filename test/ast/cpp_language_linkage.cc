// Copyright (C) 2017-2023 Jonathan Müller and cppast contributors
// SPDX-License-Identifier: MIT

#include <hercules/ast/cc/cpp_language_linkage.h>

#include <hercules/ast/cc/cpp_enum.h>

#include "test_parser.h"

using namespace hercules::ccast;

TEST_CASE("cpp_language_linkage")
{
    auto code = R"(
/// extern "C" enum a{
/// };
extern "C" enum a {};

enum b {};

/// extern "C++"{
///   enum c{
///   };
///
///   enum d{
///   };
///
///   enum e{
///   };
/// }
extern "C++" // yup
{
    enum c {};
    enum d {};
    enum e {};
}

/// extern "C++"{
/// }
extern "C++" {}

enum f {};
)";

    auto file = parse({}, "cpp_language_linkage.cpp", code);

    // check linkages
    auto count = test_visit<cpp_language_linkage>(*file, [&](const cpp_language_linkage& linkage) {
        if (linkage.name() == "\"C\"")
            REQUIRE(!linkage.is_block());
        else if (linkage.name() == "\"C++\"")
            REQUIRE(linkage.is_block());
        else
            REQUIRE(false);
    });
    REQUIRE(count == 3u);

    // check enums for their correct parent
    count = test_visit<cpp_enum>(
        *file,
        [&](const cpp_enum& e) {
            if (e.name() == "a")
                check_parent(e, "\"C\"", "a");
            else if (e.name() == "b")
                check_parent(e, "cpp_language_linkage.cpp", "b");
            else if (e.name() == "c")
                check_parent(e, "\"C++\"", "c");
            else if (e.name() == "d")
                check_parent(e, "\"C++\"", "d");
            else if (e.name() == "e")
                check_parent(e, "\"C++\"", "e");
            else if (e.name() == "f")
                check_parent(e, "cpp_language_linkage.cpp", "f");
            else
                REQUIRE(false);
        },
        false); // don't check code generation here
    REQUIRE(count == 6u);
}
