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


/*
 * Make sure the inclusion works correctly without _GNU_SOURCE
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <collie/simd/simd.h>

#include <collie/testing/doctest.h>

TEST_CASE("[GNU_SOURCE support]")
{

    SUBCASE("exp10")
    {
        CHECK_EQ(collie::simd::exp10(0.), 1.);
        CHECK_EQ(collie::simd::exp10(0.f), 1.f);
    }
}
