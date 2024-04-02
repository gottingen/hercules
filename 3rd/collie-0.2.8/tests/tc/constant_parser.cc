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
#include <collie/type_safe/detail/constant_parser.h>

#include <collie/testing/doctest.h>

using namespace collie::ts;

TEST_CASE("detail::parse")
{
    REQUIRE((detail::parse<int, '0'>() == 0));
    REQUIRE((detail::parse<int, '1', '0'>() == 10));
    REQUIRE((detail::parse<int, '4', '2', '3'>() == 423));
    REQUIRE((detail::parse<int, '2', '3', '\'', '9', '0', '0'>() == 23900));

    REQUIRE((detail::parse<int, '0', '1', '0'>() == 8));
    REQUIRE((detail::parse<int, '0', 'x', 'A'>() == 10));
    REQUIRE((detail::parse<int, '0', 'b', '1'>() == 1));
}
