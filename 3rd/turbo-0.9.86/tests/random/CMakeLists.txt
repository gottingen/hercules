#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


carbin_cc_test(
        NAME
        random_bit_gen_ref_test
        SOURCES
        "bit_gen_ref_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        random_mock_distributions_test
        SOURCES
        "mock_distributions_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_mocking_bit_gen_test
        SOURCES
        "mocking_bit_gen_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        random_bernoulli_distribution_test
        SOURCES
        "bernoulli_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)

carbin_cc_test(
        NAME
        random_beta_distribution_test
        SOURCES
        "beta_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_distributions_test
        SOURCES
        "distributions_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_generators_test
        SOURCES
        "generators_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_log_uniform_int_distribution_test
        SOURCES
        "log_uniform_int_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_discrete_distribution_test
        SOURCES
        "discrete_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        pthread
        m
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_poisson_distribution_test
        SOURCES
        "poisson_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_exponential_distribution_test
        SOURCES
        "exponential_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_gaussian_distribution_test
        SOURCES
        "gaussian_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        -lpthread
        -lm
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_uniform_int_distribution_test
        SOURCES
        "uniform_int_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
)

carbin_cc_test(
        NAME
        random_uniform_real_distribution_test
        SOURCES
        "uniform_real_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_zipf_distribution_test
        SOURCES
        "zipf_distribution_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_examples_test
        SOURCES
        "examples_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        random_seed_sequences_test
        SOURCES
        "seed_sequences_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)


# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_traits_test
        SOURCES
        "traits_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_generate_real_test
        SOURCES
        "generate_real_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        random_internal_distribution_test_util
        SOURCES
        "chi_square.cc"
        "distribution_test_util.cc"
        HEADERS
        "chi_square.h"
        "distribution_test_util.h"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${CARBIN_RANDOM_RANDEN_COPTS}
        ${CARBIN_SIMD_FLAGS}
        DEPS
        ${GTEST_LIB}
        turbo::turbo
)
# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_distribution_test_util_test
        SOURCES
        "distribution_test_util_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_fastmath_test
        SOURCES
        "fastmath_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_explicit_seed_seq_test
        SOURCES
        "explicit_seed_seq_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_salted_seed_seq_test
        SOURCES
        "salted_seed_seq_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_chi_square_test
        SOURCES
        "chi_square_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::random_internal_distribution_test_util
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_fast_uniform_bits_test
        SOURCES
        "fast_uniform_bits_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_nonsecure_base_test
        SOURCES
        "nonsecure_base_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_seed_material_test
        SOURCES
        "seed_material_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_pool_urbg_test
        SOURCES
        "pool_urbg_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_pcg_engine_test
        SOURCES
        "pcg_engine_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_randen_engine_test
        SOURCES
        "randen_engine_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_randen_test
        SOURCES
        "randen_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_randen_slow_test
        SOURCES
        "randen_slow_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_randen_hwaes_test
        SOURCES
        "randen_hwaes_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_LIB}
)


# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_uniform_helper_test
        SOURCES
        "uniform_helper_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_iostream_state_saver_test
        SOURCES
        "iostream_state_saver_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_test(
        NAME
        random_internal_wide_multiply_test
        SOURCES
        wide_multiply_test.cc
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)
carbin_cc_test(
        NAME
        exponential_biased_test
        SOURCES
        "exponential_biased_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)