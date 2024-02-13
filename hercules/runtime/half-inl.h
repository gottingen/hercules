// Copyright 2022 ByteDance Ltd. and/or its affiliates.
// Acknowledgement:
// Taken from https://github.com/pytorch/pytorch/blob/release/1.11/c10/util/Half.h
// with fixes applied:
// - change namespace to hercules::runtime for fix conflict with pytorch

#pragma once

#include <cstring>
#include <limits>

#include <hercules/runtime/runtime_port.h>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#endif

#ifdef __SYCL_DEVICE_ONLY__
#include <CL/sycl.hpp>
#endif

HERCULES_CLANG_DIAGNOSTIC_PUSH()
#if HERCULES_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
HERCULES_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace hercules {
namespace runtime {

/// Constructors

inline HERCULES_RUNTIME_HOST_DEVICE Half::Half(float value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  x = __half_as_short(__float2half(value));
#elif defined(__SYCL_DEVICE_ONLY__)
  x = sycl::bit_cast<uint16_t>(sycl::half(value));
#else
  x = detail::fp16_ieee_from_fp32_value(value);
#endif
}

/// Implicit conversions

inline HERCULES_RUNTIME_HOST_DEVICE Half::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __half2float(*reinterpret_cast<const __half*>(&x));
#elif defined(__SYCL_DEVICE_ONLY__)
  return float(sycl::bit_cast<sycl::half>(x));
#else
  return detail::fp16_ieee_to_fp32_value(x);
#endif
}

#if defined(__CUDACC__) || defined(__HIPCC__)
inline HERCULES_RUNTIME_HOST_DEVICE Half::Half(const __half& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline HERCULES_RUNTIME_HOST_DEVICE Half::operator __half() const {
  return *reinterpret_cast<const __half*>(&x);
}
#endif

// CUDA intrinsics

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)) || (defined(__clang__) && defined(__CUDA__))
inline __device__ Half __ldg(const Half* ptr) {
  return __ldg(reinterpret_cast<const __half*>(ptr));
}
#endif

/// Arithmetic

inline HERCULES_RUNTIME_HOST_DEVICE Half operator+(const Half& a, const Half& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline HERCULES_RUNTIME_HOST_DEVICE Half operator-(const Half& a, const Half& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline HERCULES_RUNTIME_HOST_DEVICE Half operator*(const Half& a, const Half& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline HERCULES_RUNTIME_HOST_DEVICE Half operator/(const Half& a, const Half& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline HERCULES_RUNTIME_HOST_DEVICE Half operator-(const Half& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || defined(__HIP_DEVICE_COMPILE__)
  return __hneg(a);
#else
  return -static_cast<float>(a);
#endif
}

inline HERCULES_RUNTIME_HOST_DEVICE Half& operator+=(Half& a, const Half& b) {
  a = a + b;
  return a;
}

inline HERCULES_RUNTIME_HOST_DEVICE Half& operator-=(Half& a, const Half& b) {
  a = a - b;
  return a;
}

inline HERCULES_RUNTIME_HOST_DEVICE Half& operator*=(Half& a, const Half& b) {
  a = a * b;
  return a;
}

inline HERCULES_RUNTIME_HOST_DEVICE Half& operator/=(Half& a, const Half& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline HERCULES_RUNTIME_HOST_DEVICE float operator+(Half a, float b) {
  return static_cast<float>(a) + b;
}
inline HERCULES_RUNTIME_HOST_DEVICE float operator-(Half a, float b) {
  return static_cast<float>(a) - b;
}
inline HERCULES_RUNTIME_HOST_DEVICE float operator*(Half a, float b) {
  return static_cast<float>(a) * b;
}
inline HERCULES_RUNTIME_HOST_DEVICE float operator/(Half a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline HERCULES_RUNTIME_HOST_DEVICE float operator+(float a, Half b) {
  return a + static_cast<float>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE float operator-(float a, Half b) {
  return a - static_cast<float>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE float operator*(float a, Half b) {
  return a * static_cast<float>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE float operator/(float a, Half b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline HERCULES_RUNTIME_HOST_DEVICE float& operator+=(float& a, const Half& b) {
  return a += static_cast<float>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE float& operator-=(float& a, const Half& b) {
  return a -= static_cast<float>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE float& operator*=(float& a, const Half& b) {
  return a *= static_cast<float>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE float& operator/=(float& a, const Half& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline HERCULES_RUNTIME_HOST_DEVICE double operator+(Half a, double b) {
  return static_cast<double>(a) + b;
}
inline HERCULES_RUNTIME_HOST_DEVICE double operator-(Half a, double b) {
  return static_cast<double>(a) - b;
}
inline HERCULES_RUNTIME_HOST_DEVICE double operator*(Half a, double b) {
  return static_cast<double>(a) * b;
}
inline HERCULES_RUNTIME_HOST_DEVICE double operator/(Half a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline HERCULES_RUNTIME_HOST_DEVICE double operator+(double a, Half b) {
  return a + static_cast<double>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE double operator-(double a, Half b) {
  return a - static_cast<double>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE double operator*(double a, Half b) {
  return a * static_cast<double>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE double operator/(double a, Half b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline HERCULES_RUNTIME_HOST_DEVICE Half operator+(Half a, int b) {
  return a + static_cast<Half>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator-(Half a, int b) {
  return a - static_cast<Half>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator*(Half a, int b) {
  return a * static_cast<Half>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator/(Half a, int b) {
  return a / static_cast<Half>(b);
}

inline HERCULES_RUNTIME_HOST_DEVICE Half operator+(int a, Half b) {
  return static_cast<Half>(a) + b;
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator-(int a, Half b) {
  return static_cast<Half>(a) - b;
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator*(int a, Half b) {
  return static_cast<Half>(a) * b;
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator/(int a, Half b) {
  return static_cast<Half>(a) / b;
}

//// Arithmetic with int64_t

inline HERCULES_RUNTIME_HOST_DEVICE Half operator+(Half a, int64_t b) {
  return a + static_cast<Half>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator-(Half a, int64_t b) {
  return a - static_cast<Half>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator*(Half a, int64_t b) {
  return a * static_cast<Half>(b);
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator/(Half a, int64_t b) {
  return a / static_cast<Half>(b);
}

inline HERCULES_RUNTIME_HOST_DEVICE Half operator+(int64_t a, Half b) {
  return static_cast<Half>(a) + b;
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator-(int64_t a, Half b) {
  return static_cast<Half>(a) - b;
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator*(int64_t a, Half b) {
  return static_cast<Half>(a) * b;
}
inline HERCULES_RUNTIME_HOST_DEVICE Half operator/(int64_t a, Half b) {
  return static_cast<Half>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Half to float.

}  // namespace runtime
}  // namespace hercules

namespace std {

template <>
class numeric_limits<::hercules::runtime::Half> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;
  static constexpr ::hercules::runtime::Half min() {
    return ::hercules::runtime::Half(0x0400, ::hercules::runtime::Half::from_bits());
  }
  static constexpr ::hercules::runtime::Half lowest() {
    return ::hercules::runtime::Half(0xFBFF, ::hercules::runtime::Half::from_bits());
  }
  static constexpr ::hercules::runtime::Half max() {
    return ::hercules::runtime::Half(0x7BFF, ::hercules::runtime::Half::from_bits());
  }
  static constexpr ::hercules::runtime::Half epsilon() {
    return ::hercules::runtime::Half(0x1400, ::hercules::runtime::Half::from_bits());
  }
  static constexpr ::hercules::runtime::Half round_error() {
    return ::hercules::runtime::Half(0x3800, ::hercules::runtime::Half::from_bits());
  }
  static constexpr ::hercules::runtime::Half infinity() {
    return ::hercules::runtime::Half(0x7C00, ::hercules::runtime::Half::from_bits());
  }
  static constexpr ::hercules::runtime::Half quiet_NaN() {
    return ::hercules::runtime::Half(0x7E00, ::hercules::runtime::Half::from_bits());
  }
  static constexpr ::hercules::runtime::Half signaling_NaN() {
    return ::hercules::runtime::Half(0x7D00, ::hercules::runtime::Half::from_bits());
  }
  static constexpr ::hercules::runtime::Half denorm_min() {
    return ::hercules::runtime::Half(0x0001, ::hercules::runtime::Half::from_bits());
  }
};

}  // namespace std

HERCULES_CLANG_DIAGNOSTIC_POP()
