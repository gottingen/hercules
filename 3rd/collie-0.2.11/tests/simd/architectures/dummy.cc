#include <collie/simd/simd.h>

// Basic check: can we instantiate a batch for the given compiler flags?
collie::simd::batch<int> come_and_get_some(collie::simd::batch<int> x, collie::simd::batch<int> y)
{
    return x + y;
}
