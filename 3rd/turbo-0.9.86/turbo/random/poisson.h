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


#ifndef TURBO_RANDOM_POISSON_H_
#define TURBO_RANDOM_POISSON_H_

#include "turbo/random/fwd.h"

namespace turbo {

    /**
     * @ingroup turbo_random_poisson
     * @brief turbo::poisson<T>(bitgen, mean = 1)
     *        turbo::poisson produces discrete probabilities for a given number of events
     *        occurring within a fixed interval within the closed interval [0, max]. `T`
     *        must be an integral type.
     *
     *        See https://en.wikipedia.org/wiki/Poisson_distribution
     *
     *        Example:
     *        @code cpp
     *        turbo::BitGen bitgen;
     *        ...
     *        int requests_per_minute = turbo::poisson<int>(bitgen, 3.2);
     *        @endcode
     * @tparam IntType
     * @tparam URBG
     * @param urbg
     * @param mean
     * @return IntType
     */
    template<typename IntType, typename URBG>
    IntType poisson(URBG &&urbg,  // NOLINT(runtime/references)
                    double mean = 1.0) {
        static_assert(random_internal::IsIntegral<IntType>::value,
                      "Template-argument 'IntType' must be an integral type, in "
                      "turbo::poisson<IntType, URBG>(...)");

        using gen_t = std::decay_t<URBG>;
        using distribution_t = typename turbo::poisson_distribution<IntType>;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, mean);
    }

    /**
     * @ingroup turbo_random_poisson
     * @brief turbo::poisson<T>(mean = 1)
     *        similar to turbo::poisson<T>(bitgen, mean = 1), using a thread-local
     *        generator.
     * @tparam IntType
     * @param mean
     * @return IntType
     */
    template<typename IntType>
    IntType poisson(double mean = 1.0) {
        static_assert(random_internal::IsIntegral<IntType>::value,
                      "Template-argument 'IntType' must be an integral type, in "
                      "turbo::poisson<IntType, URBG>(...)");

        using distribution_t = typename turbo::poisson_distribution<IntType>;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), mean);
    }

    /**
     * @ingroup turbo_random_poisson
     * @brief turbo::poisson<T>(mean = 1)
     *        similar to turbo::poisson<T>(bitgen, mean = 1), using a thread-local
     *        generator.
     * @tparam IntType
     * @param mean
     * @return IntType
     */
    template<typename IntType>
    IntType fast_poisson(double mean = 1.0) {
        static_assert(random_internal::IsIntegral<IntType>::value,
                      "Template-argument 'IntType' must be an integral type, in "
                      "turbo::poisson<IntType, URBG>(...)");

        using distribution_t = typename turbo::poisson_distribution<IntType>;

        return random_internal::DistributionCaller<InsecureBitGen>::template Call<
                distribution_t>(&get_tls_fast_bit_gen(), mean);
    }

}  // namespace turbo
#endif  // TURBO_RANDOM_POISSON_H_
