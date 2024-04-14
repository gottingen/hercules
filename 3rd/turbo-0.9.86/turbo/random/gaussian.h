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


#ifndef TURBO_RANDOM_GAUSSIAN_H_
#define TURBO_RANDOM_GAUSSIAN_H_

#include "turbo/random/fwd.h"

namespace turbo {

    /**
     * @ingroup turbo_random_gaussian
     * @brief turbo::gaussian<T>(bitgen, mean = 0, stddev = 1)
     *        turbo::gaussian produces a floating point number selected from the Gaussian
     *        (ie. "Normal") distribution. `T` must be a floating point type, but may be
     *        inferred from the types of `mean` and `stddev`.
     *        See https://en.wikipedia.org/wiki/Normal_distribution
     *
     *        Example:
     *        @code cpp
     *        turbo::BitGen bitgen;
     *        ...
     *        double giraffe_height = turbo::gaussian(bitgen, 16.3, 3.3);
     *        @endcode
     * @tparam RealType
     * @tparam URBG
     * @param urbg
     * @param mean
     * @param stddev
     * @return RealType
     */
    template<typename RealType, typename URBG>
    RealType gaussian(URBG &&urbg,  // NOLINT(runtime/references)
                      RealType mean = 0, RealType stddev = 1) {
        static_assert(
                std::is_floating_point<RealType>::value,
                "Template-argument 'RealType' must be a floating-point type, in "
                "turbo::gaussian<RealType, URBG>(...)");

        using gen_t = std::decay_t<URBG>;
        using distribution_t = typename turbo::gaussian_distribution<RealType>;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, mean, stddev);
    }

    /**
     * @ingroup turbo_random_gaussian
     * @brief  similar to turbo::gaussian<T>(bitgen, mean = 0, stddev = 1) but use the default bitgen
     *         turbo::gaussian produces a floating point number selected from the Gaussian
     *         (ie. "Normal") distribution. `T` must be a floating point type, but may be
     *         inferred from the types of `mean` and `stddev`.
     * @see   turbo::get_tls_bit_gen()
     * @see   turbo::set_tls_bit_gen()
     * @tparam RealType
     * @param mean
     * @param stddev
     * @return RealType
     */
    template<typename RealType>
    RealType gaussian(RealType mean = 0, RealType stddev = 1) {
        static_assert(
                std::is_floating_point<RealType>::value,
                "Template-argument 'RealType' must be a floating-point type, in "
                "turbo::gaussian<RealType, URBG>(...)");

        using distribution_t = typename turbo::gaussian_distribution<RealType>;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), mean, stddev);
    }

    /**
     * @ingroup turbo_random_gaussian
     * @brief  similar to turbo::gaussian<T>(bitgen, mean = 0, stddev = 1) but use the fast bitgen
     *         turbo::gaussian produces a floating point number selected from the Gaussian
     *         (ie. "Normal") distribution. `T` must be a floating point type, but may be
     *         inferred from the types of `mean` and `stddev`.
     * @see   turbo::get_tls_fast_bit_gen()
     * @see   turbo::set_tls_fast_bit_gen()
     * @tparam RealType
     * @param mean
     * @param stddev
     * @return RealType
     */
    template<typename RealType>
    RealType fast_gaussian(RealType mean = 0, RealType stddev = 1) {
        static_assert(
                std::is_floating_point<RealType>::value,
                "Template-argument 'RealType' must be a floating-point type, in "
                "turbo::gaussian<RealType, URBG>(...)");

        using distribution_t = typename turbo::gaussian_distribution<RealType>;

        return random_internal::DistributionCaller<InsecureBitGen>::template Call<
                distribution_t>(&get_tls_fast_bit_gen(), mean, stddev);
    }
}  // namespace turbo

#endif  // TURBO_RANDOM_GAUSSIAN_H_
