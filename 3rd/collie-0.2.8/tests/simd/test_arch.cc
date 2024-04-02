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
#ifndef COLLIE_SIMD_NO_SUPPORTED_ARCHITECTURE

#include <numeric>
#include <type_traits>

#include "test_sum.hpp"
#include "test_utils.hpp"

static_assert(collie::simd::default_arch::supported(), "default arch must be supported");
static_assert(std::is_same<collie::simd::default_arch, collie::simd::best_arch>::value, "default arch is the best available");
static_assert(collie::simd::supported_architectures::contains<collie::simd::default_arch>(), "default arch is supported");
static_assert(collie::simd::all_architectures::contains<collie::simd::default_arch>(), "default arch is a valid arch");

#if !COLLIE_SIMD_WITH_SVE
static_assert((std::is_same<collie::simd::default_arch, collie::simd::neon64>::value || !collie::simd::neon64::supported()), "on arm, without sve, the best we can do is neon64");
#endif

struct check_supported
{
    template <class Arch>
    void operator()(Arch) const
    {
        static_assert(Arch::supported(), "not supported?");
    }
};

struct check_available
{
    template <class Arch>
    void operator()(Arch) const
    {
        CHECK_UNARY(Arch::available());
    }
};

struct get_arch_version
{
    template <class Arch>
    unsigned operator()(Arch) { return Arch::version(); }
};

template <class T>
static bool try_load()
{
    static_assert(std::is_same<collie::simd::batch<T>, decltype(collie::simd::load_aligned(std::declval<T*>()))>::value,
                  "loading the expected type");
    static_assert(std::is_same<collie::simd::batch<T>, decltype(collie::simd::load_unaligned(std::declval<T*>()))>::value,
                  "loading the expected type");
    return true;
}

template <class... Tys>
void try_loads()
{
    (void)std::initializer_list<bool> { try_load<Tys>()... };
}

TEST_CASE("[multi arch support]")
{

    SUBCASE("collie::simd::supported_architectures")
    {
        collie::simd::supported_architectures::for_each(check_supported {});
    }

    SUBCASE("collie::simd::default_arch::name")
    {
        constexpr char const* name = collie::simd::default_arch::name();
        (void)name;
    }

    SUBCASE("collie::simd::default_arch::available")
    {
        CHECK_UNARY(collie::simd::default_arch::available());
    }

    SUBCASE("collie::simd::arch_list<...>::alignment()")
    {
        static_assert(collie::engine_list<collie::simd::generic>::alignment() == 0,
                      "generic");
        static_assert(collie::engine_list<collie::simd::sse2>::alignment()
                          == collie::simd::sse2::alignment(),
                      "one architecture");
        static_assert(collie::engine_list<collie::simd::avx512f, collie::simd::sse2>::alignment()
                          == collie::simd::avx512f::alignment(),
                      "two architectures");
    }

    SUBCASE("collie::simd::dispatch(...)")
    {
        float data[17] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f };
        float ref = std::accumulate(std::begin(data), std::end(data), 0.f);

        // platform specific
        {
            auto dispatched = collie::simd::simd_dispatch(sum {});
            float res = dispatched(data, 17);
            CHECK_EQ(ref, res);
        }

        // only highest available
        {
            auto dispatched = collie::simd::simd_dispatch<collie::engine_list<collie::simd::best_arch>>(sum {});
            float res = dispatched(data, 17);
            CHECK_EQ(ref, res);
        }

#if COLLIE_SIMD_WITH_AVX && COLLIE_SIMD_WITH_SSE2
        static_assert(collie::simd::supported_architectures::contains<collie::simd::avx>() && collie::simd::supported_architectures::contains<collie::simd::sse2>(), "consistent supported architectures");
        {
            auto dispatched = collie::simd::simd_dispatch<collie::engine_list<collie::simd::avx, collie::simd::sse2>>(sum {});
            float res = dispatched(data, 17);
            CHECK_EQ(ref, res);
        }

        // check that we pick the most appropriate version
        {
            auto dispatched = collie::simd::simd_dispatch<collie::engine_list<collie::simd::sse3, collie::simd::sse2, collie::simd::generic>>(get_arch_version {});
            unsigned expected = collie::simd::available_architectures().best >= collie::simd::sse3::version()
                ? collie::simd::sse3::version()
                : collie::simd::sse2::version();
            CHECK_EQ(expected, dispatched());
        }
#endif
    }

    SUBCASE("collie::simd::make_sized_batch_t")
    {
        using batch4f = collie::simd::make_sized_batch_t<float, 4>;
        using batch2d = collie::simd::make_sized_batch_t<double, 2>;
        using batch4i32 = collie::simd::make_sized_batch_t<int32_t, 4>;
        using batch4u32 = collie::simd::make_sized_batch_t<uint32_t, 4>;

        using batch8f = collie::simd::make_sized_batch_t<float, 8>;
        using batch4d = collie::simd::make_sized_batch_t<double, 4>;
        using batch8i32 = collie::simd::make_sized_batch_t<int32_t, 8>;
        using batch8u32 = collie::simd::make_sized_batch_t<uint32_t, 8>;

#if COLLIE_SIMD_WITH_SSE2 || COLLIE_SIMD_WITH_NEON || COLLIE_SIMD_WITH_NEON64 || COLLIE_SIMD_WITH_SVE || (COLLIE_SIMD_WITH_RVV && COLLIE_SIMD_RVV_BITS == 128)
        CHECK_EQ(4, size_t(batch4f::size));
        CHECK_EQ(4, size_t(batch4i32::size));
        CHECK_EQ(4, size_t(batch4u32::size));

        CHECK_UNARY(bool(std::is_same<float, batch4f::value_type>::value));
        CHECK_UNARY(bool(std::is_same<int32_t, batch4i32::value_type>::value));
        CHECK_UNARY(bool(std::is_same<uint32_t, batch4u32::value_type>::value));

#if COLLIE_SIMD_WITH_SSE2 || COLLIE_SIMD_WITH_NEON64 || COLLIE_SIMD_WITH_SVE || COLLIE_SIMD_WITH_RVV
        CHECK_EQ(2, size_t(batch2d::size));
        CHECK_UNARY(bool(std::is_same<double, batch2d::value_type>::value));
#else
        CHECK_UNARY(bool(std::is_same<void, batch2d>::value));
#endif

#endif
#if !COLLIE_SIMD_WITH_AVX && !COLLIE_SIMD_WITH_FMA3 && !(COLLIE_SIMD_WITH_SVE && COLLIE_SIMD_SVE_BITS == 256) && !(COLLIE_SIMD_WITH_RVV && COLLIE_SIMD_RVV_BITS == 256)
        CHECK_UNARY(bool(std::is_same<void, batch8f>::value));
        CHECK_UNARY(bool(std::is_same<void, batch4d>::value));
        CHECK_UNARY(bool(std::is_same<void, batch8i32>::value));
        CHECK_UNARY(bool(std::is_same<void, batch8u32>::value));
#else
        CHECK_EQ(8, size_t(batch8f::size));
        CHECK_EQ(8, size_t(batch8i32::size));
        CHECK_EQ(8, size_t(batch8u32::size));
        CHECK_EQ(4, size_t(batch4d::size));

        CHECK_UNARY(bool(std::is_same<float, batch8f::value_type>::value));
        CHECK_UNARY(bool(std::is_same<double, batch4d::value_type>::value));
        CHECK_UNARY(bool(std::is_same<int32_t, batch8i32::value_type>::value));
        CHECK_UNARY(bool(std::is_same<uint32_t, batch8u32::value_type>::value));
#endif
    }

    SUBCASE("collie::simd::load_(un)aligned(...) return type")
    {
        // make sure load_aligned / load_unaligned work for the default arch and
        // return the appropriate type.
        using type_list = collie::simd::mpl::type_list<short, int, long, float, std::complex<float>
#if COLLIE_SIMD_WITH_NEON64 || !COLLIE_SIMD_WITH_NEON
                                                ,
                                                double, std::complex<double>
#endif
                                                >;
        try_loads<type_list>();
    }
}

#endif
