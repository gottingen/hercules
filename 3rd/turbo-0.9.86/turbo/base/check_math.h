//
// Created by liyinbin on 2023/1/18.
//

#ifndef TURBO_BASE_CHECK_MATH_H_
#define TURBO_BASE_CHECK_MATH_H_

#include "turbo/platform/port.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef _MSC_VER
#include <intrin.h>
#endif


namespace turbo {
    namespace detail {

        template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
        bool generic_checked_add(T *result, T a, T b) {
            if (TURBO_LIKELY(a < std::numeric_limits<T>::max() - b)) {
                *result = a + b;
                return true;
            } else {
                *result = {};
                return false;
            }
        }

        template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
        bool generic_checked_small_mul(T *result, T a, T b) {
            static_assert(sizeof(T) < sizeof(uint64_t), "Too large");
            uint64_t res = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
            constexpr uint64_t overflowMask = ~((1ULL << (sizeof(T) * 8)) - 1);
            if (TURBO_UNLIKELY((res & overflowMask) != 0)) {
                *result = {};
                return false;
            }
            *result = static_cast<T>(res);
            return true;
        }

        template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
        std::enable_if_t<sizeof(T) < sizeof(uint64_t), bool> generic_checked_mul(
                T *result, T a, T b) {
            return generic_checked_small_mul(result, a, b);
        }

        template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
        std::enable_if_t<sizeof(T) == sizeof(uint64_t), bool> generic_checked_mul(
                T *result, T a, T b) {
            constexpr uint64_t halfBits = 32;
            constexpr uint64_t halfMask = (1ULL << halfBits) - 1ULL;
            uint64_t lhs_high = a >> halfBits;
            uint64_t lhs_low = a & halfMask;
            uint64_t rhs_high = b >> halfBits;
            uint64_t rhs_low = b & halfMask;

            if (TURBO_LIKELY(lhs_high == 0 && rhs_high == 0)) {
                *result = lhs_low * rhs_low;
                return true;
            }

            if (TURBO_UNLIKELY(lhs_high != 0 && rhs_high != 0)) {
                *result = {};
                return false;
            }

            uint64_t mid_bits1 = lhs_low * rhs_high;
            if (TURBO_UNLIKELY(mid_bits1 >> halfBits != 0)) {
                *result = {};
                return false;
            }

            uint64_t mid_bits2 = lhs_high * rhs_low;
            if (TURBO_UNLIKELY(mid_bits2 >> halfBits != 0)) {
                *result = {};
                return false;
            }

            uint64_t mid_bits = mid_bits1 + mid_bits2;
            if (TURBO_UNLIKELY(mid_bits >> halfBits != 0)) {
                *result = {};
                return false;
            }

            uint64_t bot_bits = lhs_low * rhs_low;
            if (TURBO_UNLIKELY(
                    !generic_checked_add(result, bot_bits, mid_bits << halfBits))) {
                *result = {};
                return false;
            }
            return true;
        }
    } // namespace detail

    template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
    bool checked_add(T *result, T a, T b) {
#if TURBO_HAVE_BUILTIN(__builtin_add_overflow)
        if (TURBO_LIKELY(!__builtin_add_overflow(a, b, result))) {
    return true;
  }
  *result = {};
  return false;
#else
        return detail::generic_checked_add(result, a, b);
#endif
    }

    template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
    bool checked_add(T *result, T a, T b, T c) {
        T tmp{};
        if (TURBO_UNLIKELY(!checked_add(&tmp, a, b))) {
            *result = {};
            return false;
        }
        if (TURBO_UNLIKELY(!checked_add(&tmp, tmp, c))) {
            *result = {};
            return false;
        }
        *result = tmp;
        return true;
    }

    template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
    bool checked_add(T *result, T a, T b, T c, T d) {
        T tmp{};
        if (TURBO_UNLIKELY(!checked_add(&tmp, a, b))) {
            *result = {};
            return false;
        }
        if (TURBO_UNLIKELY(!checked_add(&tmp, tmp, c))) {
            *result = {};
            return false;
        }
        if (TURBO_UNLIKELY(!checked_add(&tmp, tmp, d))) {
            *result = {};
            return false;
        }
        *result = tmp;
        return true;
    }

    template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
    bool checked_mul(T *result, T a, T b) {
        assert(result != nullptr);
#if TURBO_HAVE_BUILTIN(__builtin_mul_overflow)
        if (TURBO_LIKELY(!__builtin_mul_overflow(a, b, result))) {
            return true;
        }
        *result = {};
        return false;
#elif defined(_MSC_VER) && defined(TURBO_PROCESSOR_X86_64)
        static_assert(sizeof(T) <= sizeof(unsigned __int64), "Too large");
  if (sizeof(T) < sizeof(uint64_t)) {
    return detail::generic_checked_mul(result, a, b);
  } else {
    unsigned __int64 high;
    unsigned __int64 low = _umul128(a, b, &high);
    if (TURBO_LIKELY(high == 0)) {
      *result = static_cast<T>(low);
      return true;
    }
    *result = {};
    return false;
  }
#else
        return detail::generic_checked_mul(result, a, b);
#endif
    }

    template<typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
    bool checked_muladd(T *result, T base, T mul, T add) {
        T tmp{};
        if (TURBO_UNLIKELY(!checked_mul(&tmp, base, mul))) {
            *result = {};
            return false;
        }
        if (TURBO_UNLIKELY(!checked_add(&tmp, tmp, add))) {
            *result = {};
            return false;
        }
        *result = tmp;
        return true;
    }

    template<
            typename T,
            typename T2,
            typename = std::enable_if_t<std::is_pointer<T>::value>,
            typename = std::enable_if_t<std::is_unsigned<T2>::value>>
    bool checked_add(T *result, T a, T2 b) {
        return checked_muladd(
                reinterpret_cast<size_t *>(result),
                size_t(b),
                sizeof(std::remove_pointer_t<T>),
                size_t(a));
    }


    // rounds the given 64-bit unsigned integer to the nearest power of 2
    template <typename T, std::enable_if_t<
            (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 8) , void
    >* = nullptr>
    constexpr T next_pow2(T x) {
        if(x == 0) return 1;
        x--;
        x |= x>>1;
        x |= x>>2;
        x |= x>>4;
        x |= x>>8;
        x |= x>>16;
        x |= x>>32;
        x++;
        return x;
    }

// rounds the given 32-bit unsigned integer to the nearest power of 2
    template <typename T, std::enable_if_t<
            (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 4), void
    >* = nullptr>
    constexpr T next_pow2(T x) {
        if(x == 0) return 1;
        x--;
        x |= x>>1;
        x |= x>>2;
        x |= x>>4;
        x |= x>>8;
        x |= x>>16;
        x++;
        return x;
    }

    // checks if the given number if a power of 2
    template <typename T, std::enable_if_t<
            std::is_integral_v<std::decay_t<T>>, void>* = nullptr
    >
    constexpr bool is_pow2(const T& x) {
        return x && (!(x&(x-1)));
    }

    //// finds the ceil of x divided by b
    //template <typename T, std::enable_if_t<
    //  std::is_integral_v<std::decay_t<T>>, void>* = nullptr
    //>
    //constexpr T ceil(const T& x, const T& y) {
    //  //return (x + y - 1) / y;
    //  return (x-1) / y + 1;
    //}

    /**
    @brief returns floor(log2(n)), assumes n > 0
    */
    template<typename T>
    constexpr int log2(T n) {
        int log = 0;
        while (n >>= 1) {
            ++log;
        }
        return log;
    }

    /**
    @brief finds the median of three numbers of dereferenced iterators using
           the given comparator
    */
    template <typename RandItr, typename C>
    RandItr median_of_three(RandItr l, RandItr m, RandItr r, C cmp) {
        return cmp(*l, *m) ? (cmp(*m, *r) ? m : (cmp(*l, *r) ? r : l ))
                           : (cmp(*r, *m) ? m : (cmp(*r, *l) ? r : l ));
    }

    /**
    @brief finds the pseudo median of a range of items using spreaded
           nine numbers
     */
    template <typename RandItr, typename C>
    RandItr pseudo_median_of_nine(RandItr beg, RandItr end, C cmp) {
        size_t N = std::distance(beg, end);
        size_t offset = N >> 3;
        return median_of_three(
                median_of_three(beg, beg+offset, beg+(offset*2), cmp),
                median_of_three(beg+(offset*3), beg+(offset*4), beg+(offset*5), cmp),
                median_of_three(beg+(offset*6), beg+(offset*7), end-1, cmp),
                cmp
        );
    }

    /**
    @brief sorts two elements of dereferenced iterators using the given
           comparison function
    */
    template<typename Iter, typename Compare>
    void sort2(Iter a, Iter b, Compare comp) {
        if (comp(*b, *a)) std::iter_swap(a, b);
    }

    /**
    @brief sorts three elements of dereferenced iterators using the given
           comparison function
    */
    template<typename Iter, typename Compare>
    void sort3(Iter a, Iter b, Iter c, Compare comp) {
        sort2(a, b, comp);
        sort2(b, c, comp);
        sort2(a, b, comp);
    }

    /**
    @brief generates a program-wise unique id of the give type (thread-safe)
    */
    template <typename T, std::enable_if_t<std::is_integral_v<T>, void>* = nullptr>
    T unique_id() {
        static std::atomic<T> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }

    /**
    @brief updates an atomic variable with a maximum value
    */
    template <typename T>
    inline void atomic_max(std::atomic<T>& v, const T& max_v) noexcept {
        T prev = v.load(std::memory_order_relaxed);
        while(prev < max_v &&
              !v.compare_exchange_weak(prev, max_v, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
        }
    }

    /**
    @brief updates an atomic variable with a minimum value
    */
    template <typename T>
    inline void atomic_min(std::atomic<T>& v, const T& min_v) noexcept {
        T prev = v.load(std::memory_order_relaxed);
        while(prev > min_v &&
              !v.compare_exchange_weak(prev, min_v, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
        }
    }

} // namespace turbo

#endif  // TURBO_BASE_CHECK_MATH_H_
