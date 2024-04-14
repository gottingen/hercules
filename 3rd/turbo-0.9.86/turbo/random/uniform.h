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


#ifndef TURBO_RANDOM_UNIFORM_H_
#define TURBO_RANDOM_UNIFORM_H_

#include "turbo/random/fwd.h"
#include "turbo/base/internal/raw_logging.h"
#include "turbo/format/print.h"

namespace turbo {

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief produces random values of type `T` uniformly distributed in
     *        a defined interval {lo, hi}. The interval `tag` defines the type of interval
     *        which should be one of the following possible values:
     *
     *        * `turbo::IntervalOpenOpen`
     *        * `turbo::IntervalOpenClosed`
     *        * `turbo::IntervalClosedOpen`
     *        * `turbo::IntervalClosedClosed`
     *
     *        where "open" refers to an exclusive value (excluded) from the output, while
     *        "closed" refers to an inclusive value (included) from the output.
     *
     *        In the absence of an explicit return type `T`, `turbo::uniform()` will deduce
     *        the return type based on the provided endpoint arguments {A lo, B hi}.
     *        Given these endpoints, one of {A, B} will be chosen as the return type, if
     *        a type can be implicitly converted into the other in a lossless way. The
     *        lack of any such implicit conversion between {A, B} will produce a
     *        compile-time error
     *
     *        See https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
     *
     *        Example:
     *        @code  cpp
     *        turbo::BitGen bitgen;
     *
     *        // Produce a random float value between 0.0 and 1.0, inclusive
     *        auto x = turbo::uniform(turbo::IntervalClosedClosed, bitgen, 0.0f, 1.0f);
     *
     *        // The most common interval of `turbo::IntervalClosedOpen` is available by
     *        // default:
     *
     *        auto x = turbo::uniform(bitgen, 0.0f, 1.0f);
     *
     *        // Return-types are typically inferred from the arguments, however callers
     *        // can optionally provide an explicit return-type to the template.
     *
     *        auto x = turbo::uniform<float>(bitgen, 0, 1);
     *
     *        @endcode
     * @param tag defines the type of interval, which should be one of the following possible values:
     *            * `turbo::IntervalOpenOpen`
     *            * `turbo::IntervalOpenClosed`
     *            * `turbo::IntervalClosedOpen`
     *            * `turbo::IntervalClosedClosed`
     * @param urbg a uniform random bit generator
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename TagType, typename URBG>
    typename std::enable_if_t<!std::is_same<R, void>::value, R>  //
    uniform(TagType tag,
            URBG &&urbg,  // NOLINT(runtime/references)
            R lo, R hi) {
        using gen_t = std::decay_t<URBG>;
        using distribution_t = random_internal::UniformDistributionWrapper<R>;

        auto a = random_internal::uniform_lower_bound(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, tag, lo, hi);
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief  similar to `turbo::uniform(tag, urbg, lo, hi)`, but uses the default
     *         engine, which is a thread-local instance of `turbo::BitGen`
     * @param tag defines the type of interval, which should be one of the following possible values:
     *           * `turbo::IntervalOpenOpen`
     *           * `turbo::IntervalOpenClosed`
     *           * `turbo::IntervalClosedOpen`
     *           * `turbo::IntervalClosedClosed`
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @see turbo::uniform(tag, urbg, lo, hi)
     * @see get_tls_bit_gen()
     * @see set_tls_bit_gen()
     * @note This function is thread-safe.
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename TagType>
    typename std::enable_if_t<!std::is_same<R, void>::value && is_random_tag<TagType>::value, R>  //
    uniform(TagType tag,
            R lo, R hi) {
        using distribution_t = random_internal::UniformDistributionWrapper<R>;

        auto a = random_internal::uniform_lower_bound(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), tag, lo, hi);
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, but uses the default
     *        engine, which is a thread-local instance of `turbo::InsecureBitGen`
     * @param tag defines the type of interval, which should be one of the following possible values:
     *           * `turbo::IntervalOpenOpen`
     *           * `turbo::IntervalOpenClosed`
     *           * `turbo::IntervalClosedOpen`
     *           * `turbo::IntervalClosedClosed`
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @see turbo::uniform(tag, urbg, lo, hi)
     * @see get_tls_fast_bit_gen()
     * @see set_tls_fast_bit_gen()
     * @note This function is thread-safe.
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename TagType>
    typename std::enable_if_t<!std::is_same<R, void>::value && is_random_tag<TagType>::value, R>  //
    fast_uniform(TagType tag,
                 R lo, R hi) {
        using distribution_t = random_internal::UniformDistributionWrapper<R>;

        auto a = random_internal::uniform_lower_bound(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<InsecureBitGen>::template Call<
                distribution_t>(&get_tls_fast_bit_gen, tag, lo, hi);
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, but uses the default
     *         tag `turbo::IntervalClosedOpen` of [lo, hi), and returns values of type `T`
     * @tparam R the return type
     * @param urbg a uniform random bit generator
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @see turbo::uniform(tag, urbg, lo, hi)
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename URBG>
    typename std::enable_if_t<!std::is_same<R, void>::value && !is_random_tag<URBG>::value, R>  //
    uniform(URBG &&urbg,  // NOLINT(runtime/references)
            R lo, R hi) {
        using gen_t = std::decay_t<URBG>;
        using distribution_t = random_internal::UniformDistributionWrapper<R>;
        constexpr auto tag = turbo::IntervalClosedOpen;

        auto a = random_internal::uniform_lower_bound(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, lo, hi);
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, but uses the default
     *        tag `turbo::IntervalClosedOpen` of [lo, hi), and defaults engine, which is a
     *        thread-local instance of `turbo::BitGen`.
     * @tparam R the return type
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @see turbo::uniform(tag, urbg, lo, hi)
     * @see get_tls_bit_gen()
     * @see set_tls_bit_gen()
     * @note This function is thread-safe.
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void>
    typename std::enable_if_t<!std::is_same<R, void>::value, R>  //
    uniform(R lo, R hi) {
        using distribution_t = random_internal::UniformDistributionWrapper<R>;
        constexpr auto tag = turbo::IntervalClosedOpen;

        auto a = random_internal::uniform_lower_bound(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), lo, hi);
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, but uses the default
     *        tag `turbo::IntervalClosedOpen` of [lo, hi), and defaults engine, which is a
     *        thread-local instance of `turbo::InsecureBitGen`.
     * @tparam R the return type
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @see turbo::uniform(tag, urbg, lo, hi)
     * @see get_tls_fast_bit_gen()
     * @see set_tls_fast_bit_gen()
     * @note This function is thread-safe.
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void>
    typename std::enable_if_t<!std::is_same<R, void>::value, R>  //
    fast_uniform(R lo, R hi) {
        using distribution_t = random_internal::UniformDistributionWrapper<R>;
        constexpr auto tag = turbo::IntervalClosedOpen;

        auto a = random_internal::uniform_lower_bound(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<InsecureBitGen>::template
                Call<distribution_t>(&get_tls_fast_bit_gen(), lo, hi);
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, but uses  different (but compatible) lo, hi types.
     *        Note that a compile-error will result if the return type cannot be deduced correctly from the passed types.
     *        The return type is deduced from the passed types, and is the type of `lo` or `hi` which can be implicitly
     *        converted into the other in a lossless way.
     *        Example:
     *        @code  cpp
     *        turbo::BitGen bitgen;
     *        // Produce a random uint32_t and size_t type value.
     *        int32_t lo = 0;
     *        size_t hi = 100;
     *        // at this way,`R` is `void` it self. will be deduced from the passed types.
     *        // `R` is `size` type. it is a using the mist suitable type of `lo` and `hi`.
     *        // simple to say, `R` is the most big type of `lo` and `hi`.but it is not always true.
     *        // shuch under the precondition that `lo` and `hi` can be implicitly converted into the other in a lossless way.
     *        // simple to say, type A and type B should be the same type or the same series type, eg signed and unsigned.
     *        // like `int8_t` and `int16_t` can be so int8_t can be promoted to int16_t.
     *        auto x = turbo::uniform(turbo::IntervalClosedClosed, bitgen, lo, hi);
     *        @endcode
     * @tparam R the return type which is deduced from the passed types, and is the type of `lo` or `hi` which can be implicitly
     *         converted into the other in a lossless way.
     * @param tag defines the type of interval, which should be one of the following possible values:
     * @param urbg a uniform random bit generator
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     * @note make sure that `A` and `B` can be implicitly converted into the other in a lossless way.
     *       eg they are all signed or unsigned, or all the same type.
     *       Intuitive representation with code:
     *          @code
     *          std::is_same<A, B>::value ||
     *          std::is_signed<A>::value == std::is_signed<B>::value ||
     *          std::is_unsigned<A>::value == std::is_unsigned<B>::value ||
     *          std::is_float_point<A>::value == std::is_float_point<B>::value
     *          @endcode
     */
    template<typename R = void, typename TagType, typename URBG, typename A,
            typename B>
    typename std::enable_if_t<
            std::is_same<R, void>::value && is_random_tag<TagType>::value && !is_random_tag<URBG>::value,
            random_internal::uniform_inferred_return_t<A, B>>
    uniform(TagType tag,
            URBG &&urbg,  // NOLINT(runtime/references)
            A lo, B hi) {
        using gen_t = std::decay_t<URBG>;
        using return_t = typename random_internal::uniform_inferred_return_t<A, B>;
        using distribution_t = random_internal::UniformDistributionWrapper<return_t>;

        auto a = random_internal::uniform_lower_bound<return_t>(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound<return_t>(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, tag, static_cast<return_t>(lo),
                                static_cast<return_t>(hi));
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, but uses  different (but compatible) lo, hi types.
     *        uses the default engine, which is a thread-local instance of `turbo::BitGen`.
     *@see turbo::uniform(tag, urbg, lo, hi)
     * @tparam R the return type which is deduced from the passed types, and is the type of `lo` or `hi` which can be implicitly
     *        converted into the other in a lossless way.
     * @param tag defines the type of interval.
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename TagType, typename A,
            typename B>
    typename std::enable_if_t<std::is_same<R, void>::value && is_random_tag<TagType>::value,
            random_internal::uniform_inferred_return_t<A, B>>
    uniform(TagType tag,
            A lo, B hi) {
        using return_t = typename random_internal::uniform_inferred_return_t<A, B>;
        using distribution_t = random_internal::UniformDistributionWrapper<return_t>;

        auto a = random_internal::uniform_lower_bound<return_t>(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound<return_t>(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), tag, static_cast<return_t>(lo),
                                static_cast<return_t>(hi));
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, but uses  different (but compatible) lo, hi types.
     *        uses the default engine, which is a thread-local instance of `turbo::InsecureBitGen`.
     *@see turbo::uniform(tag, urbg, lo, hi)
     * @tparam R the return type which is deduced from the passed types, and is the type of `lo` or `hi` which can be implicitly
     *        converted into the other in a lossless way.
     * @param tag defines the type of interval.
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename TagType, typename A,
            typename B>
    typename std::enable_if_t<std::is_same<R, void>::value && is_random_tag<TagType>::value,
            random_internal::uniform_inferred_return_t<A, B>>
    fast_uniform(TagType tag,
                 A lo, B hi) {
        using return_t = typename random_internal::uniform_inferred_return_t<A, B>;
        using distribution_t = random_internal::UniformDistributionWrapper<return_t>;

        auto a = random_internal::uniform_lower_bound<return_t>(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound<return_t>(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<InsecureBitGen>::template Call<
                distribution_t>(&get_tls_fast_bit_gen(), tag, static_cast<return_t>(lo),
                                static_cast<return_t>(hi));
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, using default tag `turbo::IntervalClosedOpen` of [lo, hi).
     * @see turbo::uniform(tag, urbg, lo, hi)
     * @tparam R the return type which is deduced from the passed types, and is the type of `lo` or `hi` which can be implicitly
     *       converted into the other in a lossless way.
     * @param urbg a uniform random bit generator
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename URBG, typename A, typename B>
    typename std::enable_if_t<std::is_same<R, void>::value && !is_random_tag<URBG>::value,
            random_internal::uniform_inferred_return_t<A, B>>
    uniform(URBG &&urbg,  // NOLINT(runtime/references)
            A lo, B hi) {
        using gen_t = std::decay_t<URBG>;
        using return_t = typename random_internal::uniform_inferred_return_t<A, B>;
        using distribution_t = random_internal::UniformDistributionWrapper<return_t>;

        constexpr auto tag = turbo::IntervalClosedOpen;
        auto a = random_internal::uniform_lower_bound<return_t>(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound<return_t>(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg, static_cast<return_t>(lo),
                                static_cast<return_t>(hi));
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, using default tag `turbo::IntervalClosedOpen` of [lo, hi).
     *        uses the default engine, which is a thread-local instance of `turbo::BitGen`.
     * @see turbo::uniform(tag, urbg, lo, hi)
     * @see get_tls_bit_gen()
     * @see set_tls_bit_gen()
     * @note This function is thread-safe.
     * @tparam R the return type which is deduced from the passed types, and is the type of `lo` or `hi` which can be implicitly
     *       converted into the other in a lossless way.
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename A, typename B>
    typename std::enable_if_t<std::is_same<R, void>::value,
            random_internal::uniform_inferred_return_t<A, B>>
    uniform(A lo, B hi) {
        using return_t = typename random_internal::uniform_inferred_return_t<A, B>;
        using distribution_t = random_internal::UniformDistributionWrapper<return_t>;

        constexpr auto tag = turbo::IntervalClosedOpen;
        auto a = random_internal::uniform_lower_bound<return_t>(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound<return_t>(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen(), static_cast<return_t>(lo),
                                static_cast<return_t>(hi));
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, using default tag `turbo::IntervalClosedOpen` of [lo, hi).
     *        uses the default engine, which is a thread-local instance of `turbo::InsecureBitGen`.
     * @see turbo::uniform(tag, urbg, lo, hi)
     * @see get_tls_fast_bit_gen()
     * @see set_tls_fast_bit_gen()
     * @note This function is thread-safe.
     * @tparam R the return type which is deduced from the passed types, and is the type of `lo` or `hi` which can be implicitly
     *       converted into the other in a lossless way.
     * @param lo the lower bound of the interval
     * @param hi the upper bound of the interval
     * @return a random value of type `T` uniformly distributed in a defined interval {lo, hi}
     */
    template<typename R = void, typename A, typename B>
    typename std::enable_if_t<std::is_same<R, void>::value,
            random_internal::uniform_inferred_return_t<A, B>>
    fast_uniform(A lo, B hi) {
        using return_t = typename random_internal::uniform_inferred_return_t<A, B>;
        using distribution_t = random_internal::UniformDistributionWrapper<return_t>;

        constexpr auto tag = turbo::IntervalClosedOpen;
        auto a = random_internal::uniform_lower_bound<return_t>(tag, lo, hi);
        auto b = random_internal::uniform_upper_bound<return_t>(tag, lo, hi);
        if (!random_internal::is_uniform_range_valid(a, b)) return lo;

        return random_internal::DistributionCaller<InsecureBitGen>::template Call<
                distribution_t>(&get_tls_fast_bit_gen(), static_cast<return_t>(lo),
                                static_cast<return_t>(hi));
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(tag, urbg, lo, hi)`, using std::numeric_limits<T>::min() and std::numeric_limits<T>::max() as the interval.
     *        using default tag `turbo::IntervalClosedOpen` of [lo, hi).
     * @param urbg a uniform random bit generator
     * @tparam T the return type which is deduced from the passed types.
     * @return a random value of type `T` uniformly distributed in a defined interval [lo, hi)
     * @note `T` must be unsigned.
     */
    template<typename R, typename URBG>
    typename std::enable_if_t<!std::is_signed<R>::value && !is_random_tag<URBG>::value, R>  //
    uniform(URBG &&urbg) {  // NOLINT(runtime/references)
        using gen_t = std::decay_t<URBG>;
        using distribution_t = random_internal::UniformDistributionWrapper<R>;

        return random_internal::DistributionCaller<gen_t>::template Call<
                distribution_t>(&urbg);
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(urbs)`, using default engine, which is a thread-local instance of `turbo::BitGen`.
     * @tparam R the return type which is deduced from the passed types.
     * @return a random value of type `R` uniformly distributed in a defined interval [lo, hi)
     * @note `R` must be unsigned.
     */
    template<typename R>
    typename std::enable_if_t<!std::is_signed<R>::value, R>  //
    uniform() {
        using distribution_t = random_internal::UniformDistributionWrapper<R>;

        return random_internal::DistributionCaller<BitGen>::template Call<
                distribution_t>(&get_tls_bit_gen());
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief similar to `turbo::uniform(urbs)`, using default engine, which is a thread-local instance of `turbo::InsecureBitGen`.
     * @tparam R the return type which is deduced from the passed types.
     * @return a random value of type `R` uniformly distributed in a defined interval [lo, hi)
     * @note `R` must be unsigned.
     */
    template<typename R>
    typename std::enable_if_t<!std::is_signed<R>::value, R>  //
    fast_uniform() {
        using distribution_t = random_internal::UniformDistributionWrapper<R>;

        return random_internal::DistributionCaller<InsecureBitGen>::template Call<
                distribution_t>(&get_tls_fast_bit_gen());
    }

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief a tool kit class for using uniform distribution, to generate
     *        random values of type `T` uniformly distributed in a defined interval {lo, hi}.
     * @tparam T the return type which is deduced from the passed types.
     * @tparam URBG the uniform random bit generator.
     */
    template<typename T, typename URBG = BitGen>
    class FixedUniform {
    public:
        FixedUniform(T lo, T hi) : lo_(lo), hi_(hi), distribution_(lo, hi) {}

        T operator()() {
            return distribution_(urbg_);
        }

        T hi() const { return hi_; }

        T lo() const { return lo_; }

    private:
        T lo_{0};
        T hi_{0};
        URBG urbg_;
        uniform_int_distribution<T> distribution_;
    };

    /**
     * @ingroup turbo_random_uniform Turbo Random Module
     * @brief a tool kit class for using uniform distribution, to generate
     *        random values of type `T` uniformly distributed in a defined interval list.
     *        the interval list is a vector of pair of {lo, hi}.
     *        Example:
     *        @code  cpp
     *        turbo::BitGen bitgen;
     *        // Produce a random uint32_t and size_t type value.
     *        std::vector<std::pair<uint32_t, uint32_t>> ranges = {{0, 10}, {20, 30}, {40, 50}};
     *        turbo::FixedUniformRanges<int32_t, int32_t> fixed_uniform_ranges(ranges);
     *        auto x = fixed_uniform_ranges();
     *        @endcode
     * @tparam R the return type which is passed to the template.
     * @tparam T the return type which is passed to the template.
     * @tparam URBG the uniform random bit generator.
     * @note `T` and `R` must be same type or the same series type, eg signed and unsigned.
     */
    template<typename R, typename T, typename URBG = BitGen>
    class FixedUniformRanges {
    public:
        FixedUniformRanges(const std::vector<std::pair<T, T>> &ranges)
                : _urbg(), _ranges(ranges), _range_index(0, ranges.size() > 0 ? ranges.size() - 1 : 0) {
            for (auto &range: _ranges) {
                _distribution.emplace_back(range.first, range.second);
            }
        }

        R operator()() {
            auto index = _range_index(_urbg);
            return static_cast<R>(_distribution[index](_urbg));
        }

    private:
        URBG _urbg;
        std::vector<std::pair<T, T>> _ranges;
        uniform_int_distribution<size_t> _range_index;
        std::vector<uniform_int_distribution<T>> _distribution;
    };
}  // namespace turbo

#endif  // TURBO_RANDOM_UNIFORM_H_
