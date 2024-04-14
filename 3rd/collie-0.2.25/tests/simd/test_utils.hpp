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

#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#include <collie/testing/doctest.h>

#ifndef SIMD_TEST_UTILS_HPP
#define SIMD_TEST_UTILS_HPP

/**************************
 * AppleClang workarounds *
 *************************/

// AppleClang is known for having precision issues
// in the gamma function codegen. It's known to happen
// between AVX and AVX2, but it also happens on SSE4.1
// in GitHub Actions.
// This also seems to happen in M1.
struct precision_t
{
#if defined(__apple_build_version__) && (COLLIE_SIMD_WITH_SSE4_1 || COLLIE_SIMD_WITH_NEON64)
    static constexpr size_t max = 8192;
#else
    static constexpr size_t max = 2048;
#endif
};

/*******************
 * Pretty printers *
 *******************/

class simd_test_names
{
public:
    template <class T>
    static std::string GetName(int)
    {
        using value_type = typename T::value_type;
        std::string prefix;
#if COLLIE_SIMD_WITH_SSE
        size_t register_size = T::size * sizeof(value_type) * CHAR_BIT;
        if (register_size == size_t(128))
        {
            prefix = "sse_";
        }
        else if (register_size == size_t(256))
        {
            prefix = "avx_";
        }
        else if (register_size == size_t(512))
        {
            prefix = "avx512_";
        }
#elif COLLIE_SIMD_WITH_NEON
        size_t register_size = T::size * sizeof(value_type) * CHAR_BIT;
        if (register_size == size_t(128))
        {
            prefix = "arm_";
        }
#endif
        if (std::is_same<value_type, uint8_t>::value)
        {
            return prefix + "uint8_t";
        }
        if (std::is_same<value_type, int8_t>::value)
        {
            return prefix + "int8_t";
        }
        if (std::is_same<value_type, uint16_t>::value)
        {
            return prefix + "uint16_t";
        }
        if (std::is_same<value_type, int16_t>::value)
        {
            return prefix + "int16_t";
        }
        if (std::is_same<value_type, uint32_t>::value)
        {
            return prefix + "uint32_t";
        }
        if (std::is_same<value_type, int32_t>::value)
        {
            return prefix + "int32_t";
        }
        if (std::is_same<value_type, uint64_t>::value)
        {
            return prefix + "uint64_t";
        }
        if (std::is_same<value_type, int64_t>::value)
        {
            return prefix + "int64_t";
        }
        if (std::is_same<value_type, float>::value)
        {
            return prefix + "float";
        }
        if (std::is_same<value_type, double>::value)
        {
            return prefix + "double";
        }
        if (std::is_same<value_type, std::complex<float>>::value)
        {
            return prefix + "complex<float>";
        }
        if (std::is_same<value_type, std::complex<double>>::value)
        {
            return prefix + "complex<double>";
        }

        return prefix + "unknow_type";
    }
};

inline std::string print_function_name(const std::string& func)
{
    return std::string("  while testing ") + func;
}

/************************
 * Comparison functions *
 ************************/

namespace collie::simd
{
    template <class T, class A, size_t N>
    inline bool operator==(const batch<T, A>& lhs, const std::array<T, N>& rhs)
    {
        std::array<T, N> tmp;
        lhs.store_unaligned(tmp.data());
        return tmp == rhs;
    }

    template <class T, class A, size_t N>
    inline bool operator==(const std::array<T, N>& lhs, const batch<T, A>& rhs)
    {
        return rhs == lhs;
    }
}

namespace detail
{
    namespace utils
    {
        // define some overloads here as integer versions do not exist for msvc
        template <class T>
        inline typename std::enable_if<!std::is_integral<T>::value, bool>::type isinf(const T& c)
        {
            return std::isinf(c);
        }

        template <class T>
        inline typename std::enable_if<std::is_integral<T>::value, bool>::type isinf(const T&)
        {
            return false;
        }

        template <class T>
        inline typename std::enable_if<!std::is_integral<T>::value, bool>::type isnan(const T& c)
        {
            return std::isnan(c);
        }

        template <class T>
        inline typename std::enable_if<std::is_integral<T>::value, bool>::type isnan(const T&)
        {
            return false;
        }
    }

    inline unsigned char uabs(unsigned char val)
    {
        return val;
    }

    inline unsigned short uabs(unsigned short val)
    {
        return val;
    }

    inline unsigned int uabs(unsigned int val)
    {
        return val;
    }

    inline unsigned long uabs(unsigned long val)
    {
        return val;
    }

    inline unsigned long long uabs(unsigned long long val)
    {
        return val;
    }

    template <class T>
    inline T uabs(T val)
    {
        return std::abs(val);
    }

    template <class T>
    bool check_is_small(const T& value, const T& tolerance)
    {
        using std::abs;
        return uabs(value) < uabs(tolerance);
    }

    template <class T>
    T safe_division(const T& lhs, const T& rhs)
    {
        if (rhs == T(0))
        {
            return (std::numeric_limits<T>::max)();
        }
        if (rhs < static_cast<T>(1) && lhs > rhs * (std::numeric_limits<T>::max)())
        {
            return (std::numeric_limits<T>::max)();
        }
        if ((lhs == static_cast<T>(0)) || (rhs > static_cast<T>(1) && lhs < rhs * (std::numeric_limits<T>::min)()))
        {
            return static_cast<T>(0);
        }
        return lhs / rhs;
    }

    template <class T>
    bool check_is_close(const T& lhs, const T& rhs, const T& relative_precision)
    {
        using std::abs;
        T diff = uabs(lhs - rhs);
        T d1 = safe_division(diff, T(uabs(rhs)));
        T d2 = safe_division(diff, T(uabs(lhs)));

        return d1 <= relative_precision && d2 <= relative_precision;
    }

    template <class T>
    struct scalar_comparison_near
    {
        static bool run(const T& lhs, const T& rhs)
        {
            using std::abs;
            using std::max;

            // direct compare integers -- but need tolerance for inexact double conversion
            if (std::is_integral<T>::value && lhs < 10e6 && rhs < 10e6)
            {
                return lhs == rhs;
            }

            if (utils::isnan(lhs))
            {
                return utils::isnan(rhs);
            }

            if (utils::isinf(lhs))
            {
                return utils::isinf(rhs) && (lhs * rhs > 0) /* same sign */;
            }

            T relative_precision = precision_t::max * std::numeric_limits<T>::epsilon();
            T absolute_zero_prox = precision_t::max * std::numeric_limits<T>::epsilon();

            if (max(uabs(lhs), uabs(rhs)) < T(1e-3))
            {
                using res_type = decltype(lhs - rhs);
                return detail::check_is_small(lhs - rhs, res_type(absolute_zero_prox));
            }
            else
            {
                return detail::check_is_close(lhs, rhs, relative_precision);
            }
        }
    };

    template <class T>
    struct scalar_comparison
    {
        static bool run(const T& lhs, const T& rhs)
        {
            return lhs == rhs;
        }
    };

    template <>
    struct scalar_comparison<float> : scalar_comparison_near<float>
    {
    };

    template <>
    struct scalar_comparison<double> : scalar_comparison_near<double>
    {
    };

    template <class T>
    struct scalar_comparison<std::complex<T>>
    {
        static bool run(const std::complex<T>& lhs, const std::complex<T>& rhs)
        {
            using real_comparison = scalar_comparison<T>;
            return real_comparison::run(lhs.real(), rhs.real()) && real_comparison::run(lhs.imag(), rhs.imag());
        }
    };

    template <class V>
    struct vector_comparison
    {
        static bool run(const V& lhs, const V& rhs)
        {
            using value_type = typename V::value_type;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (!scalar_comparison<value_type>::run(lhs[i], rhs[i]))
                    return false;
            }
            return true;
        }
    };

    template <class T>
    bool expect_scalar_near(const T& lhs, const T& rhs)
    {
        return scalar_comparison<T>::run(lhs, rhs);
    }

    template <class V>
    bool expect_container_near(const V& lhs, const V& rhs)
    {
        return vector_comparison<V>::run(lhs, rhs);
    }

    template <class T, size_t N>
    bool expect_array_near(const std::array<T, N>& lhs, const std::array<T, N>& rhs)
    {
        return expect_container_near(lhs, rhs);
    }

    template <class T, class A>
    bool expect_vector_near(const std::vector<T, A>& lhs, const std::vector<T, A>& rhs)
    {
        return expect_container_near(lhs, rhs);
    }

    template <class T, size_t N, class A>
    bool expect_batch_near(const ::collie::simd::batch<T, A>& lhs, const std::array<T, N>& rhs)
    {
        std::array<T, N> tmp;
        lhs.store_unaligned(tmp.data());
        return expect_array_near(tmp, rhs);
    }

    template <class T, size_t N, class A>
    bool expect_batch_near(const std::array<T, N>& lhs, const ::collie::simd::batch<T, A>& rhs)
    {
        std::array<T, N> tmp;
        rhs.store_unaligned(tmp.data());
        return expect_array_near(lhs, tmp);
    }

    template <class T, class A>
    bool expect_batch_near(const ::collie::simd::batch<T, A>& lhs, const ::collie::simd::batch<T, A>& rhs)
    {
        constexpr auto N = collie::simd::batch<T, A>::size;
        std::array<T, N> tmp;
        lhs.store_unaligned(tmp.data());
        return expect_batch_near(tmp, rhs);
    }

    template <class T, size_t N, class A>
    bool expect_batch_near(const ::collie::simd::batch_bool<T, A>& lhs, const std::array<bool, N>& rhs)
    {
        std::array<bool, N> tmp;
        lhs.store_unaligned(tmp.data());
        return expect_array_near(tmp, rhs);
    }

    template <class T, size_t N, class A>
    bool expect_batch_near(const std::array<bool, N>& lhs, const ::collie::simd::batch_bool<T, A>& rhs)
    {
        std::array<bool, N> tmp;
        rhs.store_unaligned(tmp.data());
        return expect_array_near(lhs, tmp);
    }

    template <class T, class A>
    bool expect_batch_near(const ::collie::simd::batch_bool<T, A>& lhs, const ::collie::simd::batch_bool<T, A>& rhs)
    {
        constexpr auto N = collie::simd::batch<T, A>::size;
        std::array<bool, N> tmp;
        lhs.store_unaligned(tmp.data());
        return expect_batch_near(tmp, rhs);
    }

    template <class It>
    size_t get_nb_diff(It lhs_begin, It lhs_end, It rhs_begin)
    {
        size_t res = 0;
        using value_type = typename std::iterator_traits<It>::value_type;
        while (lhs_begin != lhs_end)
        {
            if (!scalar_comparison<value_type>::run(*lhs_begin++, *rhs_begin++))
            {
                ++res;
            }
        }
        return res;
    }

    template <class T, class A>
    size_t get_nb_diff(const std::vector<T, A>& lhs, const std::vector<T, A>& rhs)
    {
        return get_nb_diff(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, size_t N>
    size_t get_nb_diff(const std::array<T, N>& lhs, const std::array<T, N>& rhs)
    {
        return get_nb_diff(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, class A>
    size_t get_nb_diff_near(const std::vector<T, A>& lhs, const std::vector<T, A>& rhs, float precision)
    {
        size_t i = 0;
        for (size_t i = 0; i < lhs.size(); i++)
        {
            if (std::abs(lhs[i] - rhs[i]) > precision)
            {
                i++;
            }
        }
        return i;
    }

    template <class T, size_t N>
    size_t get_nb_diff_near(const std::array<T, N>& lhs, const std::array<T, N>& rhs, float precision)
    {
        size_t i = 0;
        for (size_t i = 0; i < lhs.size(); i++)
        {
            if (std::abs(lhs[i] - rhs[i]) > precision)
            {
                i++;
            }
        }
        return i;
    }

    template <class B, class S>
    void load_batch(B& b, const S& src, size_t i = 0)
    {
        b = B::load_unaligned(src.data() + i);
    }

    template <class B, class D>
    void store_batch(const B& b, D& dst, size_t i = 0)
    {
        b.store_unaligned(dst.data() + i);
    }

}

#define CHECK_BATCH_EQ(b1, b2)                            \
    do                                                    \
    {                                                     \
        INFO(#b1 ":", b1);                                \
        INFO(#b2 ":", b2);                                \
        CHECK_UNARY(::detail::expect_batch_near(b1, b2)); \
    } while (0)
#define CHECK_SCALAR_EQ(s1, s2)                            \
    do                                                     \
    {                                                      \
        INFO(#s1 ":", s1);                                 \
        INFO(#s2 ":", s2);                                 \
        CHECK_UNARY(::detail::expect_scalar_near(s1, s2)); \
    } while (0)
#define CHECK_VECTOR_EQ(v1, v2)                            \
    do                                                     \
    {                                                      \
        INFO(#v1 ":", v1);                                 \
        INFO(#v2 ":", v2);                                 \
        CHECK_UNARY(::detail::expect_vector_near(v1, v2)); \
    } while (0)

namespace collie::simd
{
    /************************
     * Enable metafunctions *
     ************************/

    // Backport of C++14 std::enable_if
    template <bool B, class T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;

    template <class T, class R>
    using enable_integral_t = enable_if_t<std::is_integral<T>::value, R>;

    template <class T, class R>
    using enable_floating_point_t = enable_if_t<std::is_floating_point<T>::value, R>;

    namespace mpl
    {
        /**************
         * types_list *
         **************/
        template <class... T>
        struct type_list
        {
        };
    }
}

/***********************
 * Testing types lists *
 ***********************/

#define BATCH_INT_TYPES collie::simd::batch<uint8_t>,  \
                        collie::simd::batch<int8_t>,   \
                        collie::simd::batch<uint16_t>, \
                        collie::simd::batch<int16_t>,  \
                        collie::simd::batch<uint32_t>, \
                        collie::simd::batch<int32_t>,  \
                        collie::simd::batch<uint64_t>, \
                        collie::simd::batch<int64_t>

#if COLLIE_SIMD_WITH_NEON64 || !COLLIE_SIMD_WITH_NEON
#define BATCH_FLOAT_TYPES collie::simd::batch<float>, collie::simd::batch<double>
#else
#define BATCH_FLOAT_TYPES collie::simd::batch<float>
#endif
#if COLLIE_SIMD_WITH_NEON64 || !COLLIE_SIMD_WITH_NEON
#define BATCH_COMPLEX_TYPES collie::simd::batch<std::complex<float>>, collie::simd::batch<std::complex<double>>
#else
#define BATCH_COMPLEX_TYPES collie::simd::batch<std::complex<float>>
#endif

#define BATCH_TYPES BATCH_INT_TYPES, BATCH_FLOAT_TYPES
#define BATCH_MATH_TYPES collie::simd::batch<int32_t>, BATCH_FLOAT_TYPES

#if !COLLIE_SIMD_WITH_AVX || COLLIE_SIMD_WITH_AVX2
#define BATCH_SWIZZLE_TAIL , collie::simd::batch<uint32_t>, collie::simd::batch<int32_t>, collie::simd::batch<uint64_t>, collie::simd::batch<int64_t>
#else
#define BATCH_SWIZZLE_TAIL
#endif

#define BATCH_SWIZZLE_TYPES BATCH_FLOAT_TYPES, BATCH_COMPLEX_TYPES BATCH_SWIZZLE_TAIL

/********************
 * conversion utils *
 ********************/
template <size_t N, size_t A>
struct conversion_param
{
    static constexpr size_t size = N;
    static constexpr size_t alignment = A;
};

#define CONVERSION_TYPES conversion_param<sizeof(collie::simd::types::simd_register<int, collie::simd::default_arch>) / sizeof(double), collie::simd::default_arch::alignment()>

#endif // SIMD_TEST_UTILS_HPP
