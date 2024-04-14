//
// Copyright 2020 The Turbo Authors.
//
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


#ifndef TURBO_BASE_INT128_H_
#define TURBO_BASE_INT128_H_

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <utility>

#include "turbo/platform/port.h"

#ifndef TURBO_HAVE_INTRINSIC_INT128
#error not support
#endif

namespace turbo {
    class int128;

    // uint128
    //
    // An unsigned 128-bit integer type. The API is meant to mimic an intrinsic type
    // as closely as is practical, including exhibiting undefined behavior in
    // analogous cases (e.g. division by zero). This type is intended to be a
    // drop-in replacement once C++ supports an intrinsic `uint128_t` type; when
    // that occurs, existing well-behaved uses of `uint128` will continue to work
    // using that new type.
    //
    // Note: code written with this type will continue to compile once `uint128_t`
    // is introduced, provided the replacement helper functions
    // `Uint128(Low|High)64()` and `make_uint128()` are made.
    //
    // A `uint128` supports the following:
    //
    //   * Implicit construction from integral types
    //   * Explicit conversion to integral types
    //
    // Additionally, if your compiler supports `__int128`, `uint128` is
    // interoperable with that type. (Turbo checks for this compatibility through
    // the `TURBO_HAVE_INTRINSIC_INT128` macro.)
    //
    // However, a `uint128` differs from intrinsic integral types in the following
    // ways:
    //
    //   * Errors on implicit conversions that do not preserve value (such as
    //     loss of precision when converting to float values).
    //   * Requires explicit construction from and conversion to floating point
    //     types.
    //   * Conversion to integral types requires an explicit static_cast() to
    //     mimic use of the `-Wnarrowing` compiler flag.
    //   * The alignment requirement of `uint128` may differ from that of an
    //     intrinsic 128-bit integer type depending on platform and build
    //     configuration.
    //
    // Example:
    //
    //     float y = turbo::uint128_max();  // Error. uint128 cannot be implicitly
    //                                    // converted to float.
    //
    //     turbo::uint128 v;
    //     uint64_t i = v;                         // Error
    //     uint64_t i = static_cast<uint64_t>(v);  // OK
    //
    class alignas(unsigned __int128) uint128 {
    public:
        uint128() = default;

        // Constructors from arithmetic types
        constexpr uint128(int v);                 // NOLINT(runtime/explicit)
        constexpr uint128(unsigned int v);        // NOLINT(runtime/explicit)
        constexpr uint128(long v);                // NOLINT(runtime/int)
        constexpr uint128(unsigned long v);       // NOLINT(runtime/int)
        constexpr uint128(long long v);           // NOLINT(runtime/int)
        constexpr uint128(unsigned long long v);  // NOLINT(runtime/int)
        constexpr uint128(__int128 v);           // NOLINT(runtime/explicit)
        constexpr uint128(unsigned __int128 v);  // NOLINT(runtime/explicit)
        constexpr uint128(int128 v);  // NOLINT(runtime/explicit)
        explicit uint128(float v);

        explicit uint128(double v);

        explicit uint128(long double v);

        // Assignment operators from arithmetic types
        constexpr uint128 &operator=(int v);

        constexpr uint128 &operator=(unsigned int v);

        constexpr uint128 &operator=(long v);                // NOLINT(runtime/int)
        constexpr uint128 &operator=(unsigned long v);       // NOLINT(runtime/int)
        constexpr uint128 &operator=(long long v);           // NOLINT(runtime/int)
        constexpr uint128 &operator=(unsigned long long v);  // NOLINT(runtime/int)
        constexpr uint128 &operator=(__int128 v);

        constexpr uint128 &operator=(unsigned __int128 v);

        constexpr uint128 &operator=(int128 v);

        // Conversion operators to other arithmetic types
        constexpr explicit operator bool() const;

        constexpr explicit operator char() const;

        constexpr explicit operator signed char() const;

        constexpr explicit operator unsigned char() const;

        constexpr explicit operator char16_t() const;

        constexpr explicit operator char32_t() const;

        constexpr explicit operator TURBO_WCHAR_T() const;

        constexpr explicit operator short() const;  // NOLINT(runtime/int)
        // NOLINTNEXTLINE(runtime/int)
        constexpr explicit operator unsigned short() const;

        constexpr explicit operator int() const;

        constexpr explicit operator unsigned int() const;

        constexpr explicit operator long() const;  // NOLINT(runtime/int)
        // NOLINTNEXTLINE(runtime/int)
        constexpr explicit operator unsigned long() const;

        // NOLINTNEXTLINE(runtime/int)
        constexpr explicit operator long long() const;

        // NOLINTNEXTLINE(runtime/int)
        constexpr explicit operator unsigned long long() const;

        constexpr explicit operator __int128() const;

        constexpr explicit operator unsigned __int128() const;

        explicit constexpr operator float() const;

        explicit constexpr operator double() const;

        explicit constexpr operator long double() const;

        // Trivial copy constructor, assignment operator and destructor.

        // Arithmetic operators.
        constexpr uint128 &operator+=(uint128 other);

        constexpr uint128 &operator-=(uint128 other);

        constexpr uint128 &operator*=(uint128 other);

        // Long division/modulo for uint128.
        constexpr uint128 &operator/=(uint128 other);

        constexpr uint128 &operator%=(uint128 other);

        constexpr uint128 operator++(int);

        constexpr uint128 operator--(int);

        constexpr uint128 &operator<<=(int);

        constexpr uint128 &operator>>=(int);

        constexpr uint128 &operator&=(uint128 other);

        constexpr uint128 &operator|=(uint128 other);

        constexpr uint128 &operator^=(uint128 other);

        constexpr uint128 &operator++();

        constexpr uint128 &operator--();

        constexpr uint64_t low64() const;

        constexpr uint64_t high64() const;

        // uint128_low64()
        //
        // Returns the lower 64-bit value of a `uint128` value.
        friend constexpr uint64_t uint128_low64(uint128 v);

        // uint128_high64()
        //
        // Returns the higher 64-bit value of a `uint128` value.
        friend constexpr uint64_t uint128_high64(uint128 v);

        // MakeUInt128()
        //
        // Constructs a `uint128` numeric value from two 64-bit unsigned integers.
        // Note that this factory function is the only way to construct a `uint128`
        // from integer values greater than 2^64.
        //
        // Example:
        //
        //   turbo::uint128 big = turbo::make_uint128(1, 0);
        friend constexpr uint128 make_uint128(uint64_t high, uint64_t low);

        // uint128_max()
        //
        // Returns the highest value for a 128-bit unsigned integer.
        friend constexpr uint128 uint128_max();

        // Support for turbo::Hash.
        template<typename H>
        friend H hash_value(H h, uint128 v) {
            return H::combine(std::move(h), uint128_high64(v), uint128_low64(v));
        }

    private:
        constexpr uint128(uint64_t high, uint64_t low);

        // TODO(strel) Update implementation to use __int128 once all users of
        // uint128 are fixed to not depend on alignof(uint128) == 8. Also add
        // alignas(16) to class definition to keep alignment consistent across
        // platforms.
#if TURBO_IS_LITTLE_ENDIAN
        uint64_t lo_;
        uint64_t hi_;
#elif TURBO_IS_BIG_ENDIAN
        uint64_t hi_;
        uint64_t lo_;
#else  // byte order
#error "Unsupported byte order: must be little-endian or big-endian."
#endif  // byte order
    };

    // Prefer to use the constexpr `uint128_max()`.
    //
    // TODO(turbo-team) deprecate kuint128max once migration tool is released.
    TURBO_DLL extern const uint128 kuint128max;

    // allow uint128 to be logged
    std::ostream &operator<<(std::ostream &os, uint128 v);

    // TODO(strel) add operator>>(std::istream&, uint128)

    constexpr uint128 uint128_max() {
        return uint128((std::numeric_limits<uint64_t>::max)(),
                       (std::numeric_limits<uint64_t>::max)());
    }

}  // namespace turbo

// Specialized numeric_limits for uint128.
namespace std {
    template<>
    class numeric_limits<turbo::uint128> {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = false;
        static constexpr bool is_integer = true;
        static constexpr bool is_exact = true;
        static constexpr bool has_infinity = false;
        static constexpr bool has_quiet_NaN = false;
        static constexpr bool has_signaling_NaN = false;
        static constexpr float_denorm_style has_denorm = denorm_absent;
        static constexpr bool has_denorm_loss = false;
        static constexpr float_round_style round_style = round_toward_zero;
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = true;
        static constexpr int digits = 128;
        static constexpr int digits10 = 38;
        static constexpr int max_digits10 = 0;
        static constexpr int radix = 2;
        static constexpr int min_exponent = 0;
        static constexpr int min_exponent10 = 0;
        static constexpr int max_exponent = 0;
        static constexpr int max_exponent10 = 0;
        static constexpr bool traps = numeric_limits<unsigned __int128>::traps;
        static constexpr bool tinyness_before = false;

        static constexpr turbo::uint128 (min)() { return 0; }

        static constexpr turbo::uint128 lowest() { return 0; }

        static constexpr turbo::uint128 (max)() { return turbo::uint128_max(); }

        static constexpr turbo::uint128 epsilon() { return 0; }

        static constexpr turbo::uint128 round_error() { return 0; }

        static constexpr turbo::uint128 infinity() { return 0; }

        static constexpr turbo::uint128 quiet_NaN() { return 0; }

        static constexpr turbo::uint128 signaling_NaN() { return 0; }

        static constexpr turbo::uint128 denorm_min() { return 0; }
    };
}  // namespace std

namespace turbo {

    // int128
    //
    // A signed 128-bit integer type. The API is meant to mimic an intrinsic
    // integral type as closely as is practical, including exhibiting undefined
    // behavior in analogous cases (e.g. division by zero).
    //
    // An `int128` supports the following:
    //
    //   * Implicit construction from integral types
    //   * Explicit conversion to integral types
    //
    // However, an `int128` differs from intrinsic integral types in the following
    // ways:
    //
    //   * It is not implicitly convertible to other integral types.
    //   * Requires explicit construction from and conversion to floating point
    //     types.

    // Additionally, if your compiler supports `__int128`, `int128` is
    // interoperable with that type. (Turbo checks for this compatibility through
    // the `TURBO_HAVE_INTRINSIC_INT128` macro.)
    //
    // The design goal for `int128` is that it will be compatible with a future
    // `int128_t`, if that type becomes a part of the standard.
    //
    // Example:
    //
    //     float y = turbo::int128(17);  // Error. int128 cannot be implicitly
    //                                  // converted to float.
    //
    //     turbo::int128 v;
    //     int64_t i = v;                        // Error
    //     int64_t i = static_cast<int64_t>(v);  // OK
    //
    class int128 {
    public:
        int128() = default;

        // Constructors from arithmetic types
        constexpr int128(int v);                 // NOLINT(runtime/explicit)
        constexpr int128(unsigned int v);        // NOLINT(runtime/explicit)
        constexpr int128(long v);                // NOLINT(runtime/int)
        constexpr int128(unsigned long v);       // NOLINT(runtime/int)
        constexpr int128(long long v);           // NOLINT(runtime/int)
        constexpr int128(unsigned long long v);  // NOLINT(runtime/int)
        constexpr int128(__int128 v);  // NOLINT(runtime/explicit)
        constexpr explicit int128(unsigned __int128 v);

        constexpr explicit int128(uint128 v);

        explicit int128(float v);

        explicit int128(double v);

        explicit int128(long double v);

        // Assignment operators from arithmetic types
        constexpr int128 &operator=(int v);

        constexpr int128 &operator=(unsigned int v);

        constexpr int128 &operator=(long v);                // NOLINT(runtime/int)
        constexpr int128 &operator=(unsigned long v);       // NOLINT(runtime/int)
        constexpr int128 &operator=(long long v);           // NOLINT(runtime/int)
        constexpr int128 &operator=(unsigned long long v);  // NOLINT(runtime/int)
        constexpr int128 &operator=(__int128 v);

        // Conversion operators to other arithmetic types
        constexpr explicit operator bool() const;

        constexpr explicit operator char() const;

        constexpr explicit operator signed char() const;

        constexpr explicit operator unsigned char() const;

        constexpr explicit operator char16_t() const;

        constexpr explicit operator char32_t() const;

        constexpr explicit operator TURBO_WCHAR_T() const;

        constexpr explicit operator short() const;  // NOLINT(runtime/int)
        // NOLINTNEXTLINE(runtime/int)
        constexpr explicit operator unsigned short() const;

        constexpr explicit operator int() const;

        constexpr explicit operator unsigned int() const;

        constexpr explicit operator long() const;  // NOLINT(runtime/int)
        // NOLINTNEXTLINE(runtime/int)
        constexpr explicit operator unsigned long() const;

        // NOLINTNEXTLINE(runtime/int)
        constexpr explicit operator long long() const;

        // NOLINTNEXTLINE(runtime/int)
        constexpr explicit operator unsigned long long() const;

        constexpr explicit operator __int128() const;

        constexpr explicit operator unsigned __int128() const;

        explicit constexpr operator float() const;

        explicit constexpr operator double() const;

        explicit constexpr operator long double() const;

        // Trivial copy constructor, assignment operator and destructor.

        // Arithmetic operators
        constexpr int128 &operator+=(int128 other);

        constexpr int128 &operator-=(int128 other);

        constexpr int128 &operator*=(int128 other);

        constexpr int128 &operator/=(int128 other);

        constexpr int128 &operator%=(int128 other);

        constexpr int128 operator++(int);  // postfix increment: i++
        constexpr int128 operator--(int);  // postfix decrement: i--
        constexpr int128 &operator++();    // prefix increment:  ++i
        constexpr int128 &operator--();    // prefix decrement:  --i
        constexpr int128 &operator&=(int128 other);

        constexpr int128 &operator|=(int128 other);

        constexpr int128 &operator^=(int128 other);

        constexpr int128 &operator<<=(int amount);

        constexpr int128 &operator>>=(int amount);

        constexpr uint64_t low64() const;

        constexpr int64_t high64() const;

        // int128_low64()
        //
        // Returns the lower 64-bit value of a `int128` value.
        friend constexpr uint64_t int128_low64(int128 v);

        // int128_high64()
        //
        // Returns the higher 64-bit value of a `int128` value.
        friend constexpr int64_t int128_high64(int128 v);

        // make_int128()
        //
        // Constructs a `int128` numeric value from two 64-bit integers. Note that
        // signedness is conveyed in the upper `high` value.
        //
        //   (turbo::int128(1) << 64) * high + low
        //
        // Note that this factory function is the only way to construct a `int128`
        // from integer values greater than 2^64 or less than -2^64.
        //
        // Example:
        //
        //   turbo::int128 big = turbo::make_int128(1, 0);
        //   turbo::int128 big_n = turbo::make_int128(-1, 0);
        friend constexpr int128 make_int128(int64_t high, uint64_t low);

        // int128_max()
        //
        // Returns the maximum value for a 128-bit signed integer.
        friend constexpr int128 int128_max();

        // int128_min()
        //
        // Returns the minimum value for a 128-bit signed integer.
        friend constexpr int128 int128_min();

        // Support for turbo::Hash.
        template<typename H>
        friend H hash_value(H h, int128 v) {
            return H::combine(std::move(h), int128_high64(v), int128_low64(v));
        }

    private:
        constexpr int128(int64_t high, uint64_t low);

        __int128 v_;
    };

    std::ostream &operator<<(std::ostream &os, int128 v);

    // TODO(turbo-team) add operator>>(std::istream&, int128)

    constexpr int128 int128_max() {
        return int128((std::numeric_limits<int64_t>::max)(),
                      (std::numeric_limits<uint64_t>::max)());
    }

    constexpr int128 int128_min() {
        return int128((std::numeric_limits<int64_t>::min)(), 0);
    }

}  // namespace turbo

// Specialized numeric_limits for int128.
namespace std {
    template<>
    class numeric_limits<turbo::int128> {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = true;
        static constexpr bool is_exact = true;
        static constexpr bool has_infinity = false;
        static constexpr bool has_quiet_NaN = false;
        static constexpr bool has_signaling_NaN = false;
        static constexpr float_denorm_style has_denorm = denorm_absent;
        static constexpr bool has_denorm_loss = false;
        static constexpr float_round_style round_style = round_toward_zero;
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 127;
        static constexpr int digits10 = 38;
        static constexpr int max_digits10 = 0;
        static constexpr int radix = 2;
        static constexpr int min_exponent = 0;
        static constexpr int min_exponent10 = 0;
        static constexpr int max_exponent = 0;
        static constexpr int max_exponent10 = 0;
        static constexpr bool traps = numeric_limits<__int128>::traps;
        static constexpr bool tinyness_before = false;

        static constexpr turbo::int128 (min)() { return turbo::int128_min(); }

        static constexpr turbo::int128 lowest() { return turbo::int128_min(); }

        static constexpr turbo::int128 (max)() { return turbo::int128_max(); }

        static constexpr turbo::int128 epsilon() { return 0; }

        static constexpr turbo::int128 round_error() { return 0; }

        static constexpr turbo::int128 infinity() { return 0; }

        static constexpr turbo::int128 quiet_NaN() { return 0; }

        static constexpr turbo::int128 signaling_NaN() { return 0; }

        static constexpr turbo::int128 denorm_min() { return 0; }
    };
}  // namespace std

// --------------------------------------------------------------------------
//                      Implementation details follow
// --------------------------------------------------------------------------
namespace turbo {

    constexpr uint128 make_uint128(uint64_t high, uint64_t low) {
        return uint128(high, low);
    }

    // Assignment from integer types.

    inline constexpr uint128 &uint128::operator=(int v) { return *this = uint128(v); }

    inline constexpr uint128 &uint128::operator=(unsigned int v) {
        return *this = uint128(v);
    }

    inline constexpr uint128 &uint128::operator=(long v) {  // NOLINT(runtime/int)
        return *this = uint128(v);
    }

    // NOLINTNEXTLINE(runtime/int)
    inline constexpr uint128 &uint128::operator=(unsigned long v) {
        return *this = uint128(v);
    }

    // NOLINTNEXTLINE(runtime/int)
    inline constexpr uint128 &uint128::operator=(long long v) {
        return *this = uint128(v);
    }

    // NOLINTNEXTLINE(runtime/int)
    inline constexpr uint128 &uint128::operator=(unsigned long long v) {
        return *this = uint128(v);
    }

    inline constexpr uint128 &uint128::operator=(__int128 v) {
        return *this = uint128(v);
    }

    inline constexpr uint128 &uint128::operator=(unsigned __int128 v) {
        return *this = uint128(v);
    }

    inline constexpr uint128 &uint128::operator=(int128 v) {
        return *this = uint128(v);
    }

    // Arithmetic operators.

    constexpr uint128 operator<<(uint128 lhs, int amount);

    constexpr uint128 operator>>(uint128 lhs, int amount);

    constexpr uint128 operator+(uint128 lhs, uint128 rhs);

    constexpr uint128 operator-(uint128 lhs, uint128 rhs);

    constexpr uint128 operator*(uint128 lhs, uint128 rhs);

    constexpr uint128 operator/(uint128 lhs, uint128 rhs);

    constexpr uint128 operator%(uint128 lhs, uint128 rhs);

    inline constexpr uint128 &uint128::operator<<=(int amount) {
        *this = *this << amount;
        return *this;
    }

    inline constexpr uint128 &uint128::operator>>=(int amount) {
        *this = *this >> amount;
        return *this;
    }

    inline constexpr uint128 &uint128::operator+=(uint128 other) {
        *this = *this + other;
        return *this;
    }

    inline constexpr uint128 &uint128::operator-=(uint128 other) {
        *this = *this - other;
        return *this;
    }

    constexpr uint128 &uint128::operator*=(uint128 other) {
        *this = *this * other;
        return *this;
    }

    inline constexpr uint128 &uint128::operator/=(uint128 other) {
        *this = *this / other;
        return *this;
    }

    inline constexpr uint128 &uint128::operator%=(uint128 other) {
        *this = *this % other;
        return *this;
    }

    constexpr uint64_t uint128_low64(uint128 v) { return v.lo_; }

    constexpr uint64_t uint128_high64(uint128 v) { return v.hi_; }

// Constructors from integer types.

#if TURBO_IS_LITTLE_ENDIAN

    constexpr uint128::uint128(uint64_t high, uint64_t low)
            : lo_{low}, hi_{high} {}

    constexpr uint128::uint128(int v)
            : lo_{static_cast<uint64_t>(v)},
              hi_{v < 0 ? (std::numeric_limits<uint64_t>::max)() : 0} {}

    constexpr uint128::uint128(long v)  // NOLINT(runtime/int)
            : lo_{static_cast<uint64_t>(v)},
              hi_{v < 0 ? (std::numeric_limits<uint64_t>::max)() : 0} {}

    constexpr uint128::uint128(long long v)  // NOLINT(runtime/int)
            : lo_{static_cast<uint64_t>(v)},
              hi_{v < 0 ? (std::numeric_limits<uint64_t>::max)() : 0} {}

    constexpr uint128::uint128(unsigned int v) : lo_{v}, hi_{0} {}

    // NOLINTNEXTLINE(runtime/int)
    constexpr uint128::uint128(unsigned long v) : lo_{v}, hi_{0} {}

    // NOLINTNEXTLINE(runtime/int)
    constexpr uint128::uint128(unsigned long long v) : lo_{v}, hi_{0} {}

    constexpr uint128::uint128(__int128 v)
            : lo_{static_cast<uint64_t>(v & ~uint64_t{0})},
              hi_{static_cast<uint64_t>(static_cast<unsigned __int128>(v) >> 64)} {}

    constexpr uint128::uint128(unsigned __int128 v)
            : lo_{static_cast<uint64_t>(v & ~uint64_t{0})},
              hi_{static_cast<uint64_t>(v >> 64)} {}

    constexpr uint128::uint128(int128 v)
            : lo_{int128_low64(v)}, hi_{static_cast<uint64_t>(int128_high64(v))} {}

#elif TURBO_IS_BIG_ENDIAN
    constexpr uint128::uint128(uint64_t high, uint64_t low)
        : hi_{high}, lo_{low} {}

    constexpr uint128::uint128(int v)
        : hi_{v < 0 ? (std::numeric_limits<uint64_t>::max)() : 0},
          lo_{static_cast<uint64_t>(v)} {}
    constexpr uint128::uint128(long v)  // NOLINT(runtime/int)
        : hi_{v < 0 ? (std::numeric_limits<uint64_t>::max)() : 0},
          lo_{static_cast<uint64_t>(v)} {}
    constexpr uint128::uint128(long long v)  // NOLINT(runtime/int)
        : hi_{v < 0 ? (std::numeric_limits<uint64_t>::max)() : 0},
          lo_{static_cast<uint64_t>(v)} {}

    constexpr uint128::uint128(unsigned int v) : hi_{0}, lo_{v} {}
    // NOLINTNEXTLINE(runtime/int)
    constexpr uint128::uint128(unsigned long v) : hi_{0}, lo_{v} {}
    // NOLINTNEXTLINE(runtime/int)
    constexpr uint128::uint128(unsigned long long v) : hi_{0}, lo_{v} {}

    constexpr uint128::uint128(__int128 v)
        : hi_{static_cast<uint64_t>(static_cast<unsigned __int128>(v) >> 64)},
          lo_{static_cast<uint64_t>(v & ~uint64_t{0})} {}
    constexpr uint128::uint128(unsigned __int128 v)
        : hi_{static_cast<uint64_t>(v >> 64)},
          lo_{static_cast<uint64_t>(v & ~uint64_t{0})} {}

    constexpr uint128::uint128(int128 v)
        : hi_{static_cast<uint64_t>(int128_high64(v))}, lo_{int128_low64(v)} {}

#else  // byte order
#error "Unsupported byte order: must be little-endian or big-endian."
#endif  // byte order

// Conversion operators to integer types.

    constexpr uint128::operator bool() const { return lo_ || hi_; }

    constexpr uint128::operator char() const { return static_cast<char>(lo_); }

    constexpr uint128::operator signed char() const {
        return static_cast<signed char>(lo_);
    }

    constexpr uint128::operator unsigned char() const {
        return static_cast<unsigned char>(lo_);
    }

    constexpr uint128::operator char16_t() const {
        return static_cast<char16_t>(lo_);
    }

    constexpr uint128::operator char32_t() const {
        return static_cast<char32_t>(lo_);
    }

    constexpr uint128::operator TURBO_WCHAR_T() const {
        return static_cast<TURBO_WCHAR_T>(lo_);
    }

    // NOLINTNEXTLINE(runtime/int)
    constexpr uint128::operator short() const { return static_cast<short>(lo_); }

    constexpr uint128::operator unsigned short() const {  // NOLINT(runtime/int)
        return static_cast<unsigned short>(lo_);            // NOLINT(runtime/int)
    }

    constexpr uint128::operator int() const { return static_cast<int>(lo_); }

    constexpr uint128::operator unsigned int() const {
        return static_cast<unsigned int>(lo_);
    }

    // NOLINTNEXTLINE(runtime/int)
    constexpr uint128::operator long() const { return static_cast<long>(lo_); }

    constexpr uint128::operator unsigned long() const {  // NOLINT(runtime/int)
        return static_cast<unsigned long>(lo_);            // NOLINT(runtime/int)
    }

    constexpr uint128::operator long long() const {  // NOLINT(runtime/int)
        return static_cast<long long>(lo_);            // NOLINT(runtime/int)
    }

    constexpr uint128::operator unsigned long long() const {  // NOLINT(runtime/int)
        return static_cast<unsigned long long>(lo_);            // NOLINT(runtime/int)
    }

    constexpr uint128::operator __int128() const {
        return (static_cast<__int128>(hi_) << 64) + lo_;
    }

    constexpr uint128::operator unsigned __int128() const {
        return (static_cast<unsigned __int128>(hi_) << 64) + lo_;
    }

    // Conversion operators to floating point types.

    inline constexpr uint128::operator float() const {
        return static_cast<float>(lo_) + std::ldexp(static_cast<float>(hi_), 64);
    }

    inline constexpr uint128::operator double() const {
        return static_cast<double>(lo_) + std::ldexp(static_cast<double>(hi_), 64);
    }

    inline constexpr uint128::operator long double() const {
        return static_cast<long double>(lo_) +
               std::ldexp(static_cast<long double>(hi_), 64);
    }

    // Comparison operators.

    constexpr bool operator==(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) ==
               static_cast<unsigned __int128>(rhs);
    }

    constexpr bool operator!=(uint128 lhs, uint128 rhs) { return !(lhs == rhs); }

    constexpr bool operator<(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) <
               static_cast<unsigned __int128>(rhs);
    }

    constexpr bool operator>(uint128 lhs, uint128 rhs) { return rhs < lhs; }

    constexpr bool operator<=(uint128 lhs, uint128 rhs) { return !(rhs < lhs); }

    constexpr bool operator>=(uint128 lhs, uint128 rhs) { return !(lhs < rhs); }

    // Unary operators.

    constexpr inline uint128 operator+(uint128 val) {
        return val;
    }

    constexpr inline int128 operator+(int128 val) {
        return val;
    }

    constexpr uint128 operator-(uint128 val) {
        return -static_cast<unsigned __int128>(val);
    }

    constexpr inline bool operator!(uint128 val) {
        return !static_cast<unsigned __int128>(val);
    }

    // Logical operators.

    constexpr inline uint128 operator~(uint128 val) {
        return ~static_cast<unsigned __int128>(val);
    }

    constexpr inline uint128 operator|(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) |
               static_cast<unsigned __int128>(rhs);
    }

    constexpr inline uint128 operator&(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) &
               static_cast<unsigned __int128>(rhs);
    }

    constexpr inline uint128 operator^(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) ^
               static_cast<unsigned __int128>(rhs);
    }

    inline constexpr uint128 &uint128::operator|=(uint128 other) {
        *this = *this | other;
        return *this;
    }

    inline constexpr uint128 &uint128::operator&=(uint128 other) {
        *this = *this & other;
        return *this;
    }

    inline constexpr uint128 &uint128::operator^=(uint128 other) {
        *this = *this ^ other;
        return *this;
    }

    // Arithmetic operators.

    constexpr uint128 operator<<(uint128 lhs, int amount) {
        return static_cast<unsigned __int128>(lhs) << amount;
    }

    constexpr uint128 operator>>(uint128 lhs, int amount) {
        return static_cast<unsigned __int128>(lhs) >> amount;
    }


    constexpr uint128 operator+(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) +
               static_cast<unsigned __int128>(rhs);
    }


    constexpr uint128 operator-(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) -
               static_cast<unsigned __int128>(rhs);
    }

    constexpr uint128 operator*(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) *
               static_cast<unsigned __int128>(rhs);
    }

    inline constexpr uint128 operator/(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) /
               static_cast<unsigned __int128>(rhs);
    }

    inline constexpr uint128 operator%(uint128 lhs, uint128 rhs) {
        return static_cast<unsigned __int128>(lhs) %
               static_cast<unsigned __int128>(rhs);
    }

    // Increment/decrement operators.

    inline constexpr uint128 uint128::operator++(int) {
        uint128 tmp(*this);
        *this += 1;
        return tmp;
    }

    inline constexpr uint128 uint128::operator--(int) {
        uint128 tmp(*this);
        *this -= 1;
        return tmp;
    }

    inline constexpr uint128 &uint128::operator++() {
        *this += 1;
        return *this;
    }

    inline constexpr uint128 &uint128::operator--() {
        *this -= 1;
        return *this;
    }

    constexpr uint64_t uint128::low64() const {
        return lo_;
    }

    constexpr uint64_t uint128::high64() const {
        return hi_;
    }

    constexpr int128 make_int128(int64_t high, uint64_t low) {
        return int128(high, low);
    }

    // Assignment from integer types.
    inline constexpr int128 &int128::operator=(int v) {
        return *this = int128(v);
    }

    inline constexpr int128 &int128::operator=(unsigned int v) {
        return *this = int128(v);
    }

    inline constexpr int128 &int128::operator=(long v) {  // NOLINT(runtime/int)
        return *this = int128(v);
    }

    // NOLINTNEXTLINE(runtime/int)
    inline constexpr int128 &int128::operator=(unsigned long v) {
        return *this = int128(v);
    }

    // NOLINTNEXTLINE(runtime/int)
    inline constexpr int128 &int128::operator=(long long v) {
        return *this = int128(v);
    }

    // NOLINTNEXTLINE(runtime/int)
    inline constexpr int128 &int128::operator=(unsigned long long v) {
        return *this = int128(v);
    }

    // Arithmetic operators.
    constexpr int128 operator-(int128 v);

    constexpr int128 operator+(int128 lhs, int128 rhs);

    constexpr int128 operator-(int128 lhs, int128 rhs);

    constexpr int128 operator*(int128 lhs, int128 rhs);

    constexpr int128 operator/(int128 lhs, int128 rhs);

    constexpr int128 operator%(int128 lhs, int128 rhs);

    constexpr int128 operator|(int128 lhs, int128 rhs);

    constexpr int128 operator&(int128 lhs, int128 rhs);

    constexpr int128 operator^(int128 lhs, int128 rhs);

    constexpr int128 operator<<(int128 lhs, int amount);

    constexpr int128 operator>>(int128 lhs, int amount);

    inline constexpr int128 &int128::operator+=(int128 other) {
        *this = *this + other;
        return *this;
    }

    inline constexpr int128 &int128::operator-=(int128 other) {
        *this = *this - other;
        return *this;
    }

    constexpr int128 &int128::operator*=(int128 other) {
        *this = *this * other;
        return *this;
    }

    inline constexpr int128 &int128::operator/=(int128 other) {
        *this = *this / other;
        return *this;
    }

    inline constexpr int128 &int128::operator%=(int128 other) {
        *this = *this % other;
        return *this;
    }

    inline constexpr int128 &int128::operator|=(int128 other) {
        *this = *this | other;
        return *this;
    }

    inline constexpr int128 &int128::operator&=(int128 other) {
        *this = *this & other;
        return *this;
    }

    inline constexpr int128 &int128::operator^=(int128 other) {
        *this = *this ^ other;
        return *this;
    }

    inline constexpr int128 &int128::operator<<=(int amount) {
        *this = *this << amount;
        return *this;
    }

    inline constexpr int128 &int128::operator>>=(int amount) {
        *this = *this >> amount;
        return *this;
    }


    // Forward declaration for comparison operators.
    constexpr bool operator!=(int128 lhs, int128 rhs);

    namespace int128_internal {

        // Casts from unsigned to signed while preserving the underlying binary
        // representation.
        constexpr int64_t BitCastToSigned(uint64_t v) {
            // Casting an unsigned integer to a signed integer of the same
            // width is implementation defined behavior if the source value would not fit
            // in the destination type. We step around it with a roundtrip bitwise not
            // operation to make sure this function remains constexpr. Clang, GCC, and
            // MSVC optimize this to a no-op on x86-64.
            return v & (uint64_t{1} << 63) ? ~static_cast<int64_t>(~v)
                                           : static_cast<int64_t>(v);
        }

    }  // namespace int128_internal

    constexpr uint64_t int128::low64() const {
        return static_cast<uint64_t>(v_ & ~uint64_t{0});
    }

    constexpr int64_t int128::high64() const {
        return int128_internal::BitCastToSigned(
                static_cast<uint64_t>(static_cast<unsigned __int128>(v_) >> 64));
    }

    namespace int128_internal {

        // Casts from unsigned to signed while preserving the underlying binary
        // representation.
        constexpr __int128 BitCastToSigned(unsigned __int128 v) {
            // Casting an unsigned integer to a signed integer of the same
            // width is implementation defined behavior if the source value would not fit
            // in the destination type. We step around it with a roundtrip bitwise not
            // operation to make sure this function remains constexpr. Clang and GCC
            // optimize this to a no-op on x86-64.
            return v & (static_cast<unsigned __int128>(1) << 127)
                   ? ~static_cast<__int128>(~v)
                   : static_cast<__int128>(v);
        }

    }  // namespace int128_internal


    inline constexpr int128 &int128::operator=(__int128 v) {
        v_ = v;
        return *this;
    }

    constexpr uint64_t int128_low64(int128 v) {
        return static_cast<uint64_t>(v.v_ & ~uint64_t{0});
    }

    constexpr int64_t int128_high64(int128 v) {
        // Initially cast to unsigned to prevent a right shift on a negative value.
        return int128_internal::BitCastToSigned(
                static_cast<uint64_t>(static_cast<unsigned __int128>(v.v_) >> 64));
    }

    constexpr int128::int128(int64_t high, uint64_t low)
    // Initially cast to unsigned to prevent a left shift that overflows.
            : v_(int128_internal::BitCastToSigned(static_cast<unsigned __int128>(high)
                                                          << 64) |
                 low) {}


    constexpr int128::int128(int v) : v_{v} {}

    constexpr int128::int128(long v) : v_{v} {}       // NOLINT(runtime/int)

    constexpr int128::int128(long long v) : v_{v} {}  // NOLINT(runtime/int)

    constexpr int128::int128(__int128 v) : v_{v} {}

    constexpr int128::int128(unsigned int v) : v_{v} {}

    constexpr int128::int128(unsigned long v) : v_{v} {}  // NOLINT(runtime/int)

// NOLINTNEXTLINE(runtime/int)
    constexpr int128::int128(unsigned long long v) : v_{v} {}

    constexpr int128::int128(unsigned __int128 v) : v_{static_cast<__int128>(v)} {}

    inline int128::int128(float v) {
        v_ = static_cast<__int128>(v);
    }

    inline int128::int128(double v) {
        v_ = static_cast<__int128>(v);
    }

    inline int128::int128(long double v) {
        v_ = static_cast<__int128>(v);
    }

    constexpr int128::int128(uint128 v) : v_{static_cast<__int128>(v)} {}

    constexpr int128::operator bool() const { return static_cast<bool>(v_); }

    constexpr int128::operator char() const { return static_cast<char>(v_); }

    constexpr int128::operator signed char() const {
        return static_cast<signed char>(v_);
    }

    constexpr int128::operator unsigned char() const {
        return static_cast<unsigned char>(v_);
    }

    constexpr int128::operator char16_t() const {
        return static_cast<char16_t>(v_);
    }

    constexpr int128::operator char32_t() const {
        return static_cast<char32_t>(v_);
    }

    constexpr int128::operator TURBO_WCHAR_T() const {
        return static_cast<TURBO_WCHAR_T>(v_);
    }

    constexpr int128::operator short() const {  // NOLINT(runtime/int)
        return static_cast<short>(v_);            // NOLINT(runtime/int)
    }

    constexpr int128::operator unsigned short() const {  // NOLINT(runtime/int)
        return static_cast<unsigned short>(v_);            // NOLINT(runtime/int)
    }

    constexpr int128::operator int() const {
        return static_cast<int>(v_);
    }

    constexpr int128::operator unsigned int() const {
        return static_cast<unsigned int>(v_);
    }

    constexpr int128::operator long() const {  // NOLINT(runtime/int)
        return static_cast<long>(v_);            // NOLINT(runtime/int)
    }

    constexpr int128::operator unsigned long() const {  // NOLINT(runtime/int)
        return static_cast<unsigned long>(v_);            // NOLINT(runtime/int)
    }

    constexpr int128::operator long long() const {  // NOLINT(runtime/int)
        return static_cast<long long>(v_);            // NOLINT(runtime/int)
    }

    constexpr int128::operator unsigned long long() const {  // NOLINT(runtime/int)
        return static_cast<unsigned long long>(v_);            // NOLINT(runtime/int)
    }

    constexpr int128::operator __int128() const { return v_; }

    constexpr int128::operator unsigned __int128() const {
        return static_cast<unsigned __int128>(v_);
    }

// Clang on PowerPC sometimes produces incorrect __int128 to floating point
// conversions. In that case, we do the conversion with a similar implementation
// to the conversion operators in int128_no_intrinsic.inc.
#if defined(__clang__) && !defined(__ppc64__)
    inline int128::operator float() const { return static_cast<float>(v_); }

inline int128::operator double() const { return static_cast<double>(v_); }

inline int128::operator long double() const {
  return static_cast<long double>(v_);
}

#else  // Clang on PowerPC

// Forward declaration for conversion operators to floating point types.
    constexpr int128 operator-(int128 v);

    constexpr bool operator!=(int128 lhs, int128 rhs);

    inline constexpr int128::operator float() const {
        // We must convert the absolute value and then negate as needed, because
        // floating point types are typically sign-magnitude. Otherwise, the
        // difference between the high and low 64 bits when interpreted as two's
        // complement overwhelms the precision of the mantissa.
        //
        // Also check to make sure we don't negate int128_min()
        return v_ < 0 && *this != int128_min()
               ? -static_cast<float>(-*this)
               : static_cast<float>(int128_low64(*this)) +
                 std::ldexp(static_cast<float>(int128_high64(*this)), 64);
    }

    inline constexpr int128::operator double() const {
        // See comment in int128::operator float() above.
        return v_ < 0 && *this != int128_min()
               ? -static_cast<double>(-*this)
               : static_cast<double>(int128_low64(*this)) +
                 std::ldexp(static_cast<double>(int128_high64(*this)), 64);
    }

    inline constexpr int128::operator long double() const {
        // See comment in int128::operator float() above.
        return v_ < 0 && *this != int128_min()
               ? -static_cast<long double>(-*this)
               : static_cast<long double>(int128_low64(*this)) +
                 std::ldexp(static_cast<long double>(int128_high64(*this)),
                            64);
    }

#endif  // Clang on PowerPC

// Comparison operators.

    constexpr bool operator==(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) == static_cast<__int128>(rhs);
    }

    constexpr bool operator!=(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) != static_cast<__int128>(rhs);
    }

    constexpr bool operator<(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) < static_cast<__int128>(rhs);
    }

    constexpr bool operator>(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) > static_cast<__int128>(rhs);
    }

    constexpr bool operator<=(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) <= static_cast<__int128>(rhs);
    }

    constexpr bool operator>=(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) >= static_cast<__int128>(rhs);
    }

    // Unary operators.

    constexpr int128 operator-(int128 v) { return -static_cast<__int128>(v); }

    constexpr bool operator!(int128 v) { return !static_cast<__int128>(v); }

    constexpr int128 operator~(int128 val) { return ~static_cast<__int128>(val); }

    // Arithmetic operators.

    constexpr int128 operator+(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) + static_cast<__int128>(rhs);
    }

    constexpr int128 operator-(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) - static_cast<__int128>(rhs);
    }

    constexpr int128 operator*(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
    }

    inline constexpr int128 operator/(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) / static_cast<__int128>(rhs);
    }

    inline constexpr int128 operator%(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) % static_cast<__int128>(rhs);
    }

    inline constexpr int128 int128::operator++(int) {
        int128 tmp(*this);
        ++v_;
        return tmp;
    }

    inline constexpr int128 int128::operator--(int) {
        int128 tmp(*this);
        --v_;
        return tmp;
    }

    inline constexpr int128 &int128::operator++() {
        ++v_;
        return *this;
    }

    inline constexpr int128 &int128::operator--() {
        --v_;
        return *this;
    }

    constexpr int128 operator|(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) | static_cast<__int128>(rhs);
    }

    constexpr int128 operator&(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) & static_cast<__int128>(rhs);
    }

    constexpr int128 operator^(int128 lhs, int128 rhs) {
        return static_cast<__int128>(lhs) ^ static_cast<__int128>(rhs);
    }

    constexpr int128 operator<<(int128 lhs, int amount) {
        return static_cast<__int128>(lhs) << amount;
    }

    constexpr int128 operator>>(int128 lhs, int amount) {
        return static_cast<__int128>(lhs) >> amount;
    }

}  // namespace turbo

#undef TURBO_WCHAR_T

#endif  // TURBO_BASE_INT128_H_
