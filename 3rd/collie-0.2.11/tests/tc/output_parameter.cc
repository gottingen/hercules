// Copyright 2024 The Elastic-AI Authors.
// part of Elastic AI Search
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
//
#include <collie/type_safe/output_parameter.h>

#include <collie/testing/doctest.h>
#include <string>

using namespace collie::ts;

TEST_CASE("output_parameter")
{
    SUBCASE("reference") {
        std::string output;

        output_parameter<std::string> out(output);
        SUBCASE("operator=") {
            std::string &res = (out = "abc");
            REQUIRE(&res == &output);
            REQUIRE(res == "abc");
        }
        SUBCASE("assign") {
            std::string &res = out.assign(3u, 'c');
            REQUIRE(&res == &output);
            REQUIRE(res == "ccc");
        }
        SUBCASE("multi") {
            std::string &res_a = (out = "abc");
            REQUIRE(&res_a == &output);
            REQUIRE(res_a == "abc");

            std::string &res_b = (out = "def");
            REQUIRE(&res_b == &output);
            REQUIRE(res_b == "def");

            std::string &res_c = out.assign(3u, 'c');
            REQUIRE(&res_c == &output);
            REQUIRE(res_c == "ccc");
        }
    }
    SUBCASE("deferred_construction") {
        deferred_construction<std::string> output;

        output_parameter<std::string> out(output);
        SUBCASE("operator=") {
            std::string &res = (out = "abc");
            REQUIRE(output.has_value());
            REQUIRE(&res == &output.value());
            REQUIRE(res == "abc");
        }
        SUBCASE("assign") {
            std::string &res = out.assign(3u, 'c');
            REQUIRE(output.has_value());
            REQUIRE(&res == &output.value());
            REQUIRE(res == "ccc");
        }
        SUBCASE("multi") {
            std::string &res_a = (out = "abc");
            REQUIRE(output.has_value());
            REQUIRE(&res_a == &output.value());
            REQUIRE(res_a == "abc");

            std::string &res_b = (out = "def");
            REQUIRE(&res_b == &output.value());
            REQUIRE(res_b == "def");

            std::string &res_c = out.assign(3u, 'c');
            REQUIRE(&res_c == &output.value());
            REQUIRE(res_c == "ccc");
        }
    }
}
