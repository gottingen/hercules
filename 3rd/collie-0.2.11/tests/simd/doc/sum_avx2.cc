// compile with -mavx2
#include "sum.hpp"
template float sum::operator()<collie::simd::avx2, float>(collie::simd::avx2, float const*, unsigned);
