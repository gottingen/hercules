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


#ifndef TURBO_RANDOM_ZIPF_H_
#define TURBO_RANDOM_ZIPF_H_

#include "turbo/random/fwd.h"

namespace turbo {

    /**
     * @ingroup turbo_random_zipf
     * @brief  turbo::zipf<T>(bitgen, hi = max, q = 2, v = 1)
     *         turbo::zipf produces discrete probabilities commonly used for modelling of
     *         rare events over the closed interval [0, hi]. The parameters `v` and `q`
     *         determine the skew of the distribution. `T`  must be an integral type, but
     *         may be inferred from the type of `hi`.
     *
     *         See http://mathworld.wolfram.com/ZipfDistribution.html
     *
     *         Example:
     *         @code
     *         turbo::BitGen bitgen;
     *         ...
     *         int term_rank = turbo::zipf<int>(bitgen);
     *         @endcode
     * @tparam IntType
     * @tparam URBG
     * @param urbg
     * @param hi
     * @param q = 2 default
     * @param v
     * @return IntType
     */
    template<typename IntType, typename URBG>
    IntType zipf(URBG &&urbg,  // NOLINT(runtime/references)
                 IntType hi = (std::numeric_limits<IntType>::max)(), double q = 2.0,
                 double v = 1.0) {
        static_assert(random_internal::IsIntegral<IntType>::value,
                      "Template-argument 'IntType' must be an integral type, in "
                      "turbo::zipf<IntType, URBG>(...)");

        using gen_t = std::decay_t<URBG>;
        using distribution_t = typename turbo::zipf_distribution<IntType>;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, hi, q, v);
    }

    /**
     * @brief  turbo::zipf<T>(hi = max, q = 2, v = 1)
     *         similar to turbo::zipf<T>(bitgen, hi = max, q = 2, v = 1)
     *         @see turbo::zipf<T>(bitgen, hi = max, q = 2, v = 1)
     *         using default bitgen from thread local storage
     *         @see turbo::get_tls_bit_gen()
     *         @see turbo::set_tls_bit_gen()
     * @tparam IntType default is int
     * @param hi default is max
     * @param q default is 2
     * @param v default is 1
     */
    template<typename IntType>
    IntType zipf(IntType hi = (std::numeric_limits<IntType>::max)(), double q = 2.0,
                 double v = 1.0) {
        static_assert(random_internal::IsIntegral<IntType>::value,
                      "Template-argument 'IntType' must be an integral type, in "
                      "turbo::zipf<IntType, URBG>(...)");

        using distribution_t = typename turbo::zipf_distribution<IntType>;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), hi, q, v);
    }

    /**
     * @brief  turbo::fast_zipf<T>(hi = max, q = 2, v = 1)
     *         similar to turbo::zipf<T>(bitgen, hi = max, q = 2, v = 1)
     *         @see turbo::zipf<T>(bitgen, hi = max, q = 2, v = 1)
     *         using default bitgen from thread local storage
     *         @see turbo::get_tls_fast_bit_gen()
     *         @see turbo::set_tls_fast_bit_gen()
     * @tparam IntType default is int
     * @param hi default is max
     * @param q default is 2
     * @param v default is 1
     */
    template<typename IntType>
    IntType fast_zipf(IntType hi = (std::numeric_limits<IntType>::max)(), double q = 2.0,
    double v = 1.0) {
    static_assert(random_internal::IsIntegral<IntType>::value,
                  "Template-argument 'IntType' must be an integral type, in "
                  "turbo::zipf<IntType, URBG>(...)");

    using distribution_t = typename turbo::zipf_distribution<IntType>;

    return random_internal::DistributionCaller<InsecureBitGen>::template Call<
            distribution_t>(&get_tls_fast_bit_gen(), hi, q, v);
}

}  // namespace turbo

#endif  // TURBO_RANDOM_ZIPF_H_
