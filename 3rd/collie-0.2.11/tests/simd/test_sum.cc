#include "test_sum.hpp"
#if COLLIE_SIMD_WITH_AVX
template float sum::operator()(collie::simd::avx, float const*, unsigned);
#endif
