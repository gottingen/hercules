#include <hercules/ast/cc/cpp_entity.h>
using namespace hercules::ccast;

#include "test_parser.h"
#include <iostream>

TEST_CASE("visitor_filtered")
{
    auto code = R"(
        namespace the_ns {
            class foo {
                enum inner_enum {};
                class bar {};
            };
            class one {}; class two {}; class three {};
            enum quaz {};
        }
        enum outer {};
    )";

    cpp_entity_index idx;
    auto             file             = parse(idx, "cpp_class.cpp", code);
    unsigned         filtered_count   = 0;
    auto             visitor_callback = [&](const cpp_entity&, hercules::ccast::visitor_info info) {
        if (info.event != hercules::ccast::visitor_info::container_entity_exit)
            ++filtered_count;
        return true;
    };

    constexpr auto all_node_count = 10, enum_count = 3, class_count = 5;

    SUBCASE("all nodes are visited")
    {
        filtered_count = 0;
        hercules::ccast::visit(*file, [](const cpp_entity&) { return true; }, visitor_callback);
        REQUIRE(filtered_count == all_node_count);
    }

    SUBCASE("filtered callback on both enter and exit")
    {
        filtered_count = 0;
        hercules::ccast::visit(*file, [](const cpp_entity&) { return true; },
                      [&](const cpp_entity&, hercules::ccast::visitor_info info) {
                          (void)info;
                          filtered_count++;
                          return true;
                      });
        REQUIRE(filtered_count == all_node_count * 2);
    }

    SUBCASE("whitelist")
    {
        SUBCASE("only one kind whitelisted")
        {
            filtered_count = 0;
            hercules::ccast::visit(*file, whitelist<cpp_entity_kind::enum_t>(), visitor_callback);
            REQUIRE(filtered_count == enum_count);
        }

        SUBCASE("many kinds whitelisted")
        {
            filtered_count = 0;
            hercules::ccast::visit(*file, whitelist<cpp_entity_kind::enum_t, cpp_entity_kind::class_t>(),
                          visitor_callback);
            REQUIRE(filtered_count == enum_count + class_count);
        }
    }

    SUBCASE("blacklist")
    {
        SUBCASE("only one kind blacklisted")
        {
            filtered_count = 0;
            hercules::ccast::visit(*file, blacklist<cpp_entity_kind::file_t>(), visitor_callback);
            REQUIRE(filtered_count == all_node_count - 1);
        }

        SUBCASE("many kinds blacklisted")
        {
            filtered_count = 0;
            hercules::ccast::visit(*file, blacklist<cpp_entity_kind::enum_t, cpp_entity_kind::class_t>(),
                          visitor_callback);
            REQUIRE(filtered_count == all_node_count - enum_count - class_count);
        }
    }
    SUBCASE("blacklist_and_children")
    {
        SUBCASE("only one kind blacklisted")
        {
            filtered_count = 0;
            hercules::ccast::visit(*file, blacklist_and_children<cpp_entity_kind::file_t>(),
                          visitor_callback);
            REQUIRE(filtered_count == 0);
        }

        SUBCASE("many kinds blacklisted")
        {
            filtered_count = 0;
            hercules::ccast::visit(*file,
                          blacklist_and_children<cpp_entity_kind::enum_t,
                                                 cpp_entity_kind::class_t>(),
                          visitor_callback);
            REQUIRE(filtered_count == all_node_count - enum_count - class_count);
        }
    }
}
