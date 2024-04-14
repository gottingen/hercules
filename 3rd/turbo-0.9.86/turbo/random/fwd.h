// Copyright 2020 The Turbo Authors.
//
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

#ifndef TURBO_RANDOM_FWD_H_
#define TURBO_RANDOM_FWD_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <type_traits>

#include "turbo/base/internal/inline_variable.h"
#include "turbo/random/engine.h"
#include "turbo/random/bernoulli_distribution.h"
#include "turbo/random/beta_distribution.h"
#include "turbo/random/exponential_distribution.h"
#include "turbo/random/gaussian_distribution.h"
#include "turbo/random/internal/distribution_caller.h"  // IWYU pragma: export
#include "turbo/random/internal/uniform_helper.h"  // IWYU pragma: export
#include "turbo/random/log_uniform_int_distribution.h"
#include "turbo/random/poisson_distribution.h"
#include "turbo/random/uniform_int_distribution.h"
#include "turbo/random/uniform_real_distribution.h"
#include "turbo/random/zipf_distribution.h"

namespace turbo {

    TURBO_INTERNAL_INLINE_CONSTEXPR(IntervalClosedClosedTag, IntervalClosedClosed,
                                    {});
    TURBO_INTERNAL_INLINE_CONSTEXPR(IntervalClosedClosedTag, IntervalClosed, {});
    TURBO_INTERNAL_INLINE_CONSTEXPR(IntervalClosedOpenTag, IntervalClosedOpen, {});
    TURBO_INTERNAL_INLINE_CONSTEXPR(IntervalOpenOpenTag, IntervalOpenOpen, {});
    TURBO_INTERNAL_INLINE_CONSTEXPR(IntervalOpenOpenTag, IntervalOpen, {});
    TURBO_INTERNAL_INLINE_CONSTEXPR(IntervalOpenClosedTag, IntervalOpenClosed, {});

    /**
     * @ingroup turbo_random_engine
     * @brief get_tls_bit_gen() returns a thread-local `turbo::BitGen` instance.
     *       This function is thread-safe.
     * @note user should not call this function directly, use turbo::uniform or
     *       other distribution functions instead.
     * @return a thread-local `turbo::BitGen` instance.
     */
    BitGen &get_tls_bit_gen();

    /**
     * @ingroup turbo_random_engine
     * @brief set_tls_bit_gen() sets a thread-local `turbo::BitGen` instance.
     *       This function is thread-safe.
     *       user can call this function to set a thread-local `turbo::BitGen` instance.
     *       at the beginning of a thread.
     *       Example:
     *       @code
     *       void thread_func() {
     *          turbo::BitGen gen;
     *          turbo::set_tls_bit_gen(std::move(gen));
     *          // use turbo::uniform or other distribution functions.
     *          while (!stop) {
     *              int value = turbo::uniform(0, 100);
     *              ...
     *              ...
     *          }
     *       }
     *       @endcode
     * @note if there not special requirement, user should not call this function.
     */
    void set_tls_bit_gen(BitGen &&bit_gen);

    /**
     * @ingroup turbo_random_engine
     * @brief get_tls_fast_bit_gen() returns a thread-local `turbo::InsecureBitGen` instance.
     *       This function is thread-safe.
     * @note user should not call this function directly, use turbo::uniform or
     *       other distribution functions instead.
     * @return a thread-local `turbo::InsecureBitGen` instance.
     */
    InsecureBitGen &get_tls_fast_bit_gen();

    /**
     * @ingroup turbo_random_engine
     * @brief set_tls_fast_bit_gen() sets a thread-local `turbo::InsecureBitGen` instance.
     *       This function is thread-safe.
     *       user can call this function to set a thread-local `turbo::InsecureBitGen` instance.
     *       at the beginning of a thread.
     *       Example:
     *       @code
     *       void thread_func() {
     *          turbo::InsecureBitGen gen;
     *          turbo::set_tls_fast_bit_gen(std::move(gen));
     *          // use turbo::uniform or other distribution functions.
     *          while (!stop) {
     *              int value = turbo::uniform(0, 100);
     *              ...
     *              ...
     *          }
     *       }
     *       @endcode
     * @note if there not special requirement, user should not call this function.
     */
    void set_tls_fast_bit_gen(InsecureBitGen &&bit_gen);

}  // namespace turbo

#endif  // TURBO_RANDOM_FWD_H_
