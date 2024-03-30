#include <collie/simd/simd.h>

namespace xs = collie::simd;
xs::batch<float> mean(xs::batch<float> lhs, xs::batch<float> rhs)
{
    return (lhs + rhs) / 2;
}
