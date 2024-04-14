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


#ifndef TURBO_RANDOM_BERNOULLI_H_
#define TURBO_RANDOM_BERNOULLI_H_

#include "turbo/random/fwd.h"

namespace turbo {

    /**
     * @ingroup turbo_random_bernoulli
     * @brief turbo::bernoulli(bitgen, p)
     *         turbo::bernoulli produces a random boolean value, with probability p
     *         (where 0.0 <= p <= 1.0) equaling true.
     *
     *         Prefer turbo::bernoulli to produce boolean values over other alternatives
     *         such as comparing an turbo::uniform() value to a specific output.
     *
     *         See https://en.wikipedia.org/wiki/Bernoulli_distribution
     *
     *         Example:
     *         @code cpp
     *         turbo::BitGen bitgen;
     *         ...
     *         if (turbo::bernoulli(bitgen, 1.0/3721.0)) {
     *              std::cout << "Asteroid field navigation successful.";
     *         }
     *         @endcode
     * @tparam URBG random generator
     * @param urbg random generator
     * @param p probability
     * @return bool
     */
    template<typename URBG>
    bool bernoulli(URBG &&urbg,  // NOLINT(runtime/references)
                   double p) {
        using gen_t = std::decay_t<URBG>;
        using distribution_t = turbo::bernoulli_distribution;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, p);
    }

    /**
     * @ingroup turbo_random_bernoulli
     * @brief turbo::bernoulli(p) is similar to turbo::bernoulli(bitgen, p)
     *        but uses a thread-local random generator for efficiency and
     *        convenience.
     *        @see turbo::bernoulli(bitgen, p)
     *        @see turbo::get_tls_bit_gen()
     *        @see turbo::set_tls_bit_gen()
     * @note This function is thread-safe.
     * @param p probability
     * @return bool
     */
    inline bool bernoulli(double p) {
        using distribution_t = turbo::bernoulli_distribution;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), p);
    }

    /**
     * @ingroup turbo_random_bernoulli
     * @brief turbo::fast_bernoulli(p) is similar to turbo::bernoulli(bitgen, p)
     *        but uses a thread-local insecure random generator for efficiency and
     *        convenience. in this case, the random generator is faster than
     *        turbo::bernoulli(p).
     *       @see turbo::bernoulli(bitgen, p)
     *       @see turbo::get_tls_fast_bit_gen()
     *       @see turbo::set_tls_fast_bit_gen()
     * @note This function is thread-safe.
     * @param p probability
     * @return bool
     */
    inline bool fast_bernoulli(double p) {
        using distribution_t = turbo::bernoulli_distribution;

        return random_internal::DistributionCaller<InsecureBitGen>::template Call<distribution_t>(
                &get_tls_fast_bit_gen(), p);
    }

}  // namespace turbo

#endif  // TURBO_RANDOM_BERNOULLI_H_
