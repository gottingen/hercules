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
struct exponential_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type exp_input;
    vector_type log_input;
    vector_type expected;
    vector_type res;

    exponential_test()
    {
        nb_input = size * 10000;
        exp_input.resize(nb_input);
        log_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            exp_input[i] = value_type(-1.5) + i * value_type(3) / nb_input;
            log_input[i] = value_type(0.001 + i * 100 / nb_input);
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_exponential_functions()
    {
        // exp
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::exp(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, exp_input, i);
                out = exp(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            INFO("exp");
            CHECK_EQ(diff, 0);
        }

        // exp2
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::exp2(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, exp_input, i);
                out = exp2(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            INFO("exp2");
            CHECK_EQ(diff, 0);
        }

        // exp10
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                           /* imprecise but enough for testing version of exp10 */
                           [](const value_type& v)
                           { return exp(log(10) * v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, exp_input, i);
                out = exp10(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            INFO("exp10");
            CHECK_EQ(diff, 0);
        }

        // expm1
        {
            std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::expm1(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, exp_input, i);
                out = expm1(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            INFO("expm1");
            CHECK_EQ(diff, 0);
        }
    }

    void test_log_functions()
    {
        // log
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::log(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, log_input, i);
                out = log(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            INFO("log");
            CHECK_EQ(diff, 0);
        }

        // log2
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::log2(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, log_input, i);
                out = log2(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            INFO("log2");
            CHECK_EQ(diff, 0);
        }

        // log10
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::log10(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, log_input, i);
                out = log10(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            INFO("log10");
            CHECK_EQ(diff, 0);
        }

        // log1p
        {
            std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::log1p(v); });
            batch_type in, out;
            for (size_t i = 0; i < nb_input; i += size)
            {
                detail::load_batch(in, log_input, i);
                out = log1p(in);
                detail::store_batch(out, res, i);
            }
            size_t diff = detail::get_nb_diff(res, expected);
            CHECK_EQ(diff, 0);
        }
    }
};

TEST_CASE_TEMPLATE("[exponential]", B, BATCH_FLOAT_TYPES)
{
    exponential_test<B> Test;

    SUBCASE("exp")
    {
        Test.test_exponential_functions();
    }

    SUBCASE("log")
    {
        Test.test_log_functions();
    }
}
#endif
