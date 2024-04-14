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
//
// Created by jeff on 23-12-9.
//

#ifndef TURBO_RANDOM_EXPONENTIAL_H_
#define TURBO_RANDOM_EXPONENTIAL_H_

#include "turbo/random/fwd.h"

namespace turbo {

    /**
     * @ingroup turbo_random_exponential
     * @brief turbo::exponential<T>(bitgen, lambda = 1)
     *        turbo::exponential produces a floating point number representing the
     *        distance (time) between two consecutive events in a point process of events
     *        occurring continuously and independently at a constant average rate. `T` must
     *        be a floating point type, but may be inferred from the type of `lambda`.
     *        See https://en.wikipedia.org/wiki/Exponential_distribution.
     *
     *        Example:
     *        @code cpp
     *        turbo::BitGen bitgen;
     *        ...
     *        double call_length = turbo::exponential(bitgen, 7.0);
     *        @endcode
     * @tparam RealType
     * @tparam URBG
     * @param urbg
     * @param lambda
     * @return RealType
     */
    template<typename RealType, typename URBG>
    RealType exponential(URBG &&urbg,  // NOLINT(runtime/references)
                         RealType lambda = 1) {
        static_assert(
                std::is_floating_point<RealType>::value,
                "Template-argument 'RealType' must be a floating-point type, in "
                "turbo::exponential<RealType, URBG>(...)");

        using gen_t = std::decay_t<URBG>;
        using distribution_t = typename turbo::exponential_distribution<RealType>;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, lambda);
    }

    /**
     * @ingroup turbo_random_exponential
     * @brief  similar to turbo::exponential<T>(bitgen, lambda = 1) but use the default bitgen
     * @tparam RealType
     * @param lambda
     * @return RealType
     */
    template<typename RealType>
    RealType exponential(RealType lambda = 1) {
        static_assert(
                std::is_floating_point<RealType>::value,
                "Template-argument 'RealType' must be a floating-point type, in "
                "turbo::exponential<RealType, URBG>(...)");

        using distribution_t = typename turbo::exponential_distribution<RealType>;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), lambda);
    }

    /**
     * @ingroup turbo_random_exponential
     * @brief  similar to turbo::exponential<T>(bitgen, lambda = 1) but use the fast bitgen
     * @tparam RealType
     * @param lambda
     * @return RealType
     */
    template<typename RealType>
            RealType fast_exponential(RealType lambda = 1) {
    static_assert(
            std::is_floating_point<RealType>::value,
            "Template-argument 'RealType' must be a floating-point type, in "
            "turbo::exponential<RealType, URBG>(...)");

    using distribution_t = typename turbo::exponential_distribution<RealType>;

    return random_internal::DistributionCaller<InsecureBitGen>::template Call<
            distribution_t>(&get_tls_fast_bit_gen(), lambda);
}

}  // namespace turbo

#endif  // TURBO_RANDOM_EXPONENTIAL_H_
