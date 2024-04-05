// compile with -msse2
#include "sum.hpp"
template float sum::operator()<collie::simd::sse2, float>(collie::simd::sse2, float const*, unsigned);
