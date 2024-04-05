#include <collie/simd/simd.h>

namespace xs = collie::simd;
template <class T, class Arch>
xs::batch<T, Arch> mean(xs::batch<T, Arch> lhs, xs::batch<T, Arch> rhs)
{
    return (lhs + rhs) / 2;
}
