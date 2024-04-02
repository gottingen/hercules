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


#include <collie/simd/simd.h>
#ifndef COLLIE_SIMD_NO_SUPPORTED_ARCHITECTURE

#include "test_utils.hpp"

template <class B>
struct select_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type lhs_input;
    vector_type rhs_input;
    vector_type expected;
    vector_type res;

    select_test()
    {
        nb_input = size * 10000;
        lhs_input.resize(nb_input);
        rhs_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            lhs_input[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            rhs_input[i] = value_type(10.2) / (i + 2) + value_type(0.25);
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_select_dynamic()
    {
        for (size_t i = 0; i < nb_input; ++i)
        {
            expected[i] = lhs_input[i] > value_type(3) ? lhs_input[i] : rhs_input[i];
        }

        batch_type lhs_in, rhs_in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_input, i);
            detail::load_batch(rhs_in, rhs_input, i);
            out = collie::simd::select(lhs_in > value_type(3), lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }
    struct pattern
    {
        static constexpr bool get(std::size_t i, std::size_t) { return i % 2; }
    };

    void test_select_static()
    {
        constexpr auto mask = collie::simd::make_batch_bool_constant<batch_type, pattern>();

        for (size_t i = 0; i < nb_input; ++i)
        {
            expected[i] = mask.get(i % size) ? lhs_input[i] : rhs_input[i];
        }

        batch_type lhs_in, rhs_in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_input, i);
            detail::load_batch(rhs_in, rhs_input, i);
            out = collie::simd::select(mask, lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }
};

TEST_CASE_TEMPLATE("[select]", B, BATCH_TYPES)
{
    select_test<B> Test;
    SUBCASE("select_dynamic") { Test.test_select_dynamic(); }
    SUBCASE("select_static") { Test.test_select_static(); }
}
#endif
