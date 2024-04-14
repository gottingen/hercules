#ifndef _SUM_HPP
#define _SUM_HPP
#include <collie/simd/simd.h>

// functor with a call method that depends on `Arch`
struct sum
{
    // It's critical not to use an in-class definition here.
    // In-class and inline definition bypass extern template mechanism.
    template <class Arch, class T>
    T operator()(Arch, T const* data, unsigned size);
};

template <class Arch, class T>
T sum::operator()(Arch, T const* data, unsigned size)
{
    using batch = collie::simd::batch<T, Arch>;
    batch acc(static_cast<T>(0));
    const unsigned n = size / batch::size * batch::size;
    for (unsigned i = 0; i != n; i += batch::size)
        acc += batch::load_unaligned(data + i);
    T star_acc = collie::simd::reduce_add(acc);
    for (unsigned i = n; i < size; ++i)
        star_acc += data[i];
    return star_acc;
}

// Inform the compiler that sse2 and avx2 implementation are to be found in another compilation unit.
extern template float sum::operator()<collie::simd::avx2, float>(collie::simd::avx2, float const*, unsigned);
extern template float sum::operator()<collie::simd::sse2, float>(collie::simd::sse2, float const*, unsigned);
#endif
