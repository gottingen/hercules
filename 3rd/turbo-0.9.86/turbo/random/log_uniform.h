// Copyright 2023 The Elastic-AI Authors.
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


#ifndef TURBO_RANDOM_LOG_UNIFORM_H_
#define TURBO_RANDOM_LOG_UNIFORM_H_

#include "turbo/random/fwd.h"

namespace turbo {

    /**
     * @ingroup turbo_random_log_uniform
     * @brief turbo::log_uniform<T>(bitgen, lo, hi, base = 2)
     *        turbo::log_uniform produces random values distributed where the log to a
     *        given base of all values is uniform in a closed interval [lo, hi]. `T` must
     *        be an integral type, but may be inferred from the types of `lo` and `hi`.
     *
     *        I.e., `Loguniform(0, n, b)` is uniformly distributed across buckets
     *        [0], [1, b-1], [b, b^2-1] .. [b^(k-1), (b^k)-1] .. [b^floor(log(n, b)), n]
     *        and is uniformly distributed within each bucket.
     *
     *        The resulting probability density is inversely related to bucket size, though
     *        values in the final bucket may be more likely than previous values. (In the
     *        extreme case where n = b^i the final value will be tied with zero as the most
     *        probable result.
     *
     *        If `lo` is nonzero then this distribution is shifted to the desired interval,
     *        so Loguniform(lo, hi, b) is equivalent to Loguniform(0, hi-lo, b)+lo.
     *
     *        See http://ecolego.facilia.se/ecolego/show/Log-Uniform%20Distribution
     *
     *        Example:
     *        @code cpp
     *        turbo::BitGen bitgen;
     *        ...
     *        int v = turbo::log_uniform(bitgen, 0, 1000);
     *        @endcode
     * @tparam IntType
     * @tparam URBG
     * @param urbg
     * @param lo
     * @param hi
     * @param base
     * @return IntType
     */
    template<typename IntType, typename URBG>
    IntType log_uniform(URBG &&urbg,  // NOLINT(runtime/references)
                       IntType lo, IntType hi, IntType base = 2) {
        static_assert(random_internal::IsIntegral<IntType>::value,
                      "Template-argument 'IntType' must be an integral type, in "
                      "turbo::log_uniform<IntType, URBG>(...)");

        using gen_t = std::decay_t<URBG>;
        using distribution_t = typename turbo::log_uniform_int_distribution<IntType>;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, lo, hi, base);
    }

    /**
     * @brief  similar to turbo::log_uniform<T>(bitgen, lo, hi, base = 2) but use the default bitgen
     * @note  the default bitgen is thread_local. so it is thread safe.
     * @see   turbo::get_tls_bit_gen()
     * @see   turbo::set_tls_bit_gen()
     * @tparam IntType
     * @param lo
     * @param hi
     * @param base
     * @return IntType
     */
    template<typename IntType>
    IntType log_uniform(IntType lo, IntType hi, IntType base = 2) {
        static_assert(random_internal::IsIntegral<IntType>::value,
                      "Template-argument 'IntType' must be an integral type, in "
                      "turbo::log_uniform<IntType, URBG>(...)");

        using distribution_t = typename turbo::log_uniform_int_distribution<IntType>;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), lo, hi, base);
    }

    /**
     * @brief  similar to turbo::log_uniform<T>(bitgen, lo, hi, base = 2) but use the fast bitgen
     * @note  the fast bitgen is thread_local. so it is thread safe.
     * @see   turbo::get_tls_fast_bit_gen()
     * @see   turbo::set_tls_fast_bit_gen()
     * @tparam IntType
     * @param lo
     * @param hi
     * @param base
     * @return IntType
     */
    template<typename IntType>
    IntType fast_log_uniform(IntType lo, IntType hi, IntType base = 2) {
        static_assert(random_internal::IsIntegral<IntType>::value,
                      "Template-argument 'IntType' must be an integral type, in "
                      "turbo::log_uniform<IntType, URBG>(...)");

        using distribution_t = typename turbo::log_uniform_int_distribution<IntType>;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_fast_bit_gen(), lo, hi, base);
    }

}  // namespace turbo

#endif // TURBO_RANDOM_LOG_UNIFORM_H_
