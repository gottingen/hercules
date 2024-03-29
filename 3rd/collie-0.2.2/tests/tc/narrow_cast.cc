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
#include <collie/type_safe/narrow_cast.h>

#include <collie/testing/doctest.h>

using namespace collie::ts;

TEST_CASE("narrow_cast<integer>")
{
    integer<int> a(4);

    integer<short> b = narrow_cast<short>(a);
    REQUIRE(static_cast<short>(b) == 4);

    integer<short> c = narrow_cast<integer<short>>(a);
    REQUIRE(static_cast<short>(c) == 4);

    integer<short> d = narrow_cast<integer<short>>(42);
    REQUIRE(static_cast<short>(d) == 42);
}

TEST_CASE("narrow_cast<floating_point>")
{
    floating_point<double> a(1.);

    floating_point<float> b = narrow_cast<float>(a);
    REQUIRE(static_cast<float>(b) == 1.);

    floating_point<float> c = narrow_cast<floating_point<float>>(a);
    REQUIRE(static_cast<float>(c) == 1.);
}
