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
struct complex_power_test
{
    using batch_type = B;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;
    using real_vector_type = std::vector<real_value_type>;

    size_t nb_input;
    vector_type lhs_nn;
    vector_type lhs_pn;
    vector_type lhs_np;
    vector_type lhs_pp;
    vector_type rhs;
    vector_type expected;
    vector_type res;

    complex_power_test()
    {
        nb_input = 10000 * size;
        lhs_nn.resize(nb_input);
        lhs_pn.resize(nb_input);
        lhs_np.resize(nb_input);
        lhs_pp.resize(nb_input);
        rhs.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            real_value_type real = (real_value_type(i) / 4 + real_value_type(1.2) * std::sqrt(real_value_type(i + 0.25))) / 100;
            real_value_type imag = (real_value_type(i) / 7 + real_value_type(1.7) * std::sqrt(real_value_type(i + 0.37))) / 100;
            lhs_nn[i] = value_type(-real, -imag);
            lhs_pn[i] = value_type(real, -imag);
            lhs_np[i] = value_type(-real, imag);
            lhs_pp[i] = value_type(real, imag);
            rhs[i] = value_type(real_value_type(10.2) / (i + 2) + real_value_type(0.25),
                                real_value_type(9.1) / (i + 3) + real_value_type(0.45));
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_abs()
    {
        real_vector_type real_expected(nb_input), real_res(nb_input);
        std::transform(lhs_np.cbegin(), lhs_np.cend(), real_expected.begin(),
                       [](const value_type& v)
                       { using std::abs; return abs(v); });
        batch_type in;
        real_batch_type out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, lhs_np, i);
            out = abs(in);
            detail::store_batch(out, real_res, i);
        }
        size_t diff = detail::get_nb_diff(real_res, real_expected);
        CHECK_EQ(diff, 0);
    }

    void test_arg()
    {
        real_vector_type real_expected(nb_input), real_res(nb_input);
        std::transform(lhs_np.cbegin(), lhs_np.cend(), real_expected.begin(),
                       [](const value_type& v)
                       { using std::arg; return arg(v); });
        batch_type in;
        real_batch_type out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, lhs_np, i);
            out = arg(in);
            detail::store_batch(out, real_res, i);
        }
        size_t diff = detail::get_nb_diff(real_res, real_expected);
        CHECK_EQ(diff, 0);
    }

    void test_pow()
    {
        test_conditional_pow<real_value_type>();
    }

    void test_sqrt_nn()
    {
        std::transform(lhs_nn.cbegin(), lhs_nn.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sqrt; return sqrt(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, lhs_nn, i);
            out = sqrt(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_sqrt_pn()
    {
        std::transform(lhs_pn.cbegin(), lhs_pn.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sqrt; return sqrt(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, lhs_pn, i);
            out = sqrt(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_sqrt_np()
    {
        std::transform(lhs_np.cbegin(), lhs_np.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sqrt; return sqrt(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, lhs_np, i);
            out = sqrt(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_sqrt_pp()
    {
        std::transform(lhs_pp.cbegin(), lhs_pp.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sqrt; return sqrt(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, lhs_pp, i);
            out = sqrt(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

private:
    void test_pow_impl()
    {
        std::transform(lhs_np.cbegin(), lhs_np.cend(), rhs.cbegin(), expected.begin(),
                       [](const value_type& l, const value_type& r)
                       { using std::pow; return pow(l, r); });
        batch_type lhs_in, rhs_in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_np, i);
            detail::load_batch(rhs_in, rhs, i);
            out = pow(lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    template <class T, typename std::enable_if<!std::is_same<T, float>::value, int>::type = 0>
    void test_conditional_pow()
    {
        test_pow_impl();
    }

    template <class T, typename std::enable_if<std::is_same<T, float>::value, int>::type = 0>
    void test_conditional_pow()
    {

#if (COLLIE_SIMD_X86_INSTR_SET >= COLLIE_SIMD_X86_AVX512_VERSION) || (COLLIE_SIMD_ARM_INSTR_SET >= COLLIE_SIMD_ARM7_NEON_VERSION)
#if DEBUG_ACCURACY
        test_pow_impl();
#endif
#else
        test_pow_impl();
#endif
    }
};

TEST_CASE_TEMPLATE("[complex power]", B, BATCH_COMPLEX_TYPES)
{
    complex_power_test<B> Test;
    SUBCASE("abs")
    {
        Test.test_abs();
    }

    SUBCASE("arg")
    {
        Test.test_arg();
    }

    SUBCASE("pow")
    {
        Test.test_pow();
    }

    SUBCASE("sqrt_nn")
    {
        Test.test_sqrt_nn();
    }

    SUBCASE("sqrt_pn")
    {
        Test.test_sqrt_pn();
    }

    SUBCASE("sqrt_np")
    {
        Test.test_sqrt_np();
    }

    SUBCASE("sqrt_pp")
    {
        Test.test_sqrt_pp();
    }
}
#endif
