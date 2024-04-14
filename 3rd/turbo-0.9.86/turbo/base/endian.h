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

#ifndef TURBO_BASE_INTERNAL_ENDIAN_H_
#define TURBO_BASE_INTERNAL_ENDIAN_H_

#include <cstdint>
#include <cstdlib>

#include "turbo/base/casts.h"
#include "turbo/platform/port.h"
#include "turbo/platform/internal/unaligned_access.h"
#include "turbo/platform/port.h"


namespace turbo {

    inline uint64_t gbswap_64(uint64_t host_int) {
#if TURBO_HAVE_BUILTIN(__builtin_bswap64) || defined(__GNUC__)
        return __builtin_bswap64(host_int);
#elif defined(_MSC_VER)
        return _byteswap_uint64(host_int);
#else
        return (((host_int & uint64_t{0xFF}) << 56) |
                ((host_int & uint64_t{0xFF00}) << 40) |
                ((host_int & uint64_t{0xFF0000}) << 24) |
                ((host_int & uint64_t{0xFF000000}) << 8) |
                ((host_int & uint64_t{0xFF00000000}) >> 8) |
                ((host_int & uint64_t{0xFF0000000000}) >> 24) |
                ((host_int & uint64_t{0xFF000000000000}) >> 40) |
                ((host_int & uint64_t{0xFF00000000000000}) >> 56));
#endif
    }

    inline uint32_t gbswap_32(uint32_t host_int) {
#if TURBO_HAVE_BUILTIN(__builtin_bswap32) || defined(__GNUC__)
        return __builtin_bswap32(host_int);
#elif defined(_MSC_VER)
        return _byteswap_ulong(host_int);
#else
        return (((host_int & uint32_t{0xFF}) << 24) |
                ((host_int & uint32_t{0xFF00}) << 8) |
                ((host_int & uint32_t{0xFF0000}) >> 8) |
                ((host_int & uint32_t{0xFF000000}) >> 24));
#endif
    }

    inline uint16_t gbswap_16(uint16_t host_int) {
#if TURBO_HAVE_BUILTIN(__builtin_bswap16) || defined(__GNUC__)
        return __builtin_bswap16(host_int);
#elif defined(_MSC_VER)
        return _byteswap_ushort(host_int);
#else
        return (((host_int & uint16_t{0xFF}) << 8) |
                ((host_int & uint16_t{0xFF00}) >> 8));
#endif
    }

    enum class EndianNess {
        SYS_LITTLE_ENDIAN,
        SYS_BIG_ENDIAN
    };


#if TURBO_IS_LITTLE_ENDIAN

    /**
     * @ingroup turbo_base_endian
     * @brief Determine if the current host is little-endian.
     */

    static constexpr bool kIsLittleEndian = true;

    static constexpr EndianNess kEndianNess = EndianNess::SYS_LITTLE_ENDIAN;
    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 16-bit quantity from host byte order to network byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint16_t ghtons(uint16_t x) { return gbswap_16(x); }

    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 32-bit quantity from host byte order to network byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint32_t ghtonl(uint32_t x) { return gbswap_32(x); }

    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 64-bit quantity from host byte order to network byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint64_t ghtonll(uint64_t x) { return gbswap_64(x); }

#elif TURBO_IS_BIG_ENDIAN

    /**
     * @ingroup turbo_base_endian
     * @brief Determine if the current host is little-endian.
     * @return True if the host is little-endian, false otherwise
     */
    static constexpr bool kIsLittleEndian = false;

    static constexpr EndianNess kEndianNess = EndianNess::SYS_BIG_ENDIAN;

    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 16-bit quantity from host byte order to network byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint16_t ghtons(uint16_t x) { return x; }

    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 32-bit quantity from host byte order to network byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint32_t ghtonl(uint32_t x) { return x; }

    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 64-bit quantity from host byte order to network byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint64_t ghtonll(uint64_t x) { return x; }

#else
#error \
    "Unsupported byte order: Either TURBO_IS_BIG_ENDIAN or " \
       "TURBO_IS_LITTLE_ENDIAN must be defined"
#endif  // byte order

    /**
     * @ingroup turbo_base_endian
     * @brief Determine if the given endian is big-endian.
     * @param e The endian to check
     * @return True if the endian is big-endian, false otherwise
     */
    constexpr bool is_big_endian(EndianNess e) {
        return e == EndianNess::SYS_BIG_ENDIAN;
    }

    /**
     * @ingroup turbo_base_endian
     * @brief Determine if the given endian is little-endian.
     * @param e The endian to check
     * @return True if the endian is little-endian, false otherwise
     */
    constexpr bool is_little_endian(EndianNess e) {
        return e == EndianNess::SYS_LITTLE_ENDIAN;
    }

    /**
     * @ingroup turbo_base_endian
     * @brief Determine if the given endian is the same as the current host.
     * @param e The endian to check
     * @return True if the endian is the same as the current host, false otherwise
     */
    constexpr bool match_system(EndianNess e) {
        return e == kEndianNess;
    }

    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 16-bit quantity from network byte order to host byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint16_t gntohs(uint16_t x) { return ghtons(x); }

    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 32-bit quantity from network byte order to host byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint32_t gntohl(uint32_t x) { return ghtonl(x); }

    /**
     * @ingroup turbo_base_endian
     * @brief Convert a 64-bit quantity from network byte order to host byte order.
     * @param x The value to convert
     * @return The converted value
     */
    inline uint64_t gntohll(uint64_t x) { return ghtonll(x); }

    // Utilities to convert numbers between the current hosts's native byte
    // order and little-endian byte order
    //
    // Load/Store methods are alignment safe
    namespace little_endian {
    // Conversion functions.
#if TURBO_IS_LITTLE_ENDIAN

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t from_host16(uint16_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t to_host16(uint16_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t from_host32(uint32_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t to_host32(uint32_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t from_host64(uint64_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t to_host64(uint64_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Determine if the current host is little-endian.
         * @return True if the host is little-endian, false otherwise
         */
        inline constexpr bool is_little_endian() { return true; }

#elif TURBO_IS_BIG_ENDIAN
        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t from_host16(uint16_t x) { return gbswap_16(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t to_host16(uint16_t x) { return gbswap_16(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t from_host32(uint32_t x) { return gbswap_32(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t to_host32(uint32_t x) { return gbswap_32(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t from_host64(uint64_t x) { return gbswap_64(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t to_host64(uint64_t x) { return gbswap_64(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Determine if the current host is little-endian.
         * @return True if the host is little-endian, false otherwise
         */
        inline constexpr bool is_little_endian() { return false; }

#endif /* ENDIAN */

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 8-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint8_t from_host(uint8_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t from_host(uint16_t x) { return from_host16(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t from_host(uint32_t x) { return from_host32(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t from_host(uint64_t x) { return from_host64(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 8-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint8_t to_host(uint8_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t to_host(uint16_t x) { return to_host16(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t to_host(uint32_t x) { return to_host32(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t to_host(uint64_t x) { return to_host64(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 8-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int8_t from_host(int8_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int16_t from_host(int16_t x) {
            return bit_cast<int16_t>(from_host16(bit_cast<uint16_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int32_t from_host(int32_t x) {
            return bit_cast<int32_t>(from_host32(bit_cast<uint32_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from host byte order to little-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int64_t from_host(int64_t x) {
            return bit_cast<int64_t>(from_host64(bit_cast<uint64_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 8-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int8_t to_host(int8_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int16_t to_host(int16_t x) {
            return bit_cast<int16_t>(to_host16(bit_cast<uint16_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int32_t to_host(int32_t x) {
            return bit_cast<int32_t>(to_host32(bit_cast<uint32_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from little-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int64_t to_host(int64_t x) {
            return bit_cast<int64_t>(to_host64(bit_cast<uint64_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Load a 16-bit quantity from a byte array even if it's unaligned.
         * @param p The address to load from
         * @return The converted value
         */
        inline uint16_t load16(const void *p) {
            return to_host16(TURBO_INTERNAL_UNALIGNED_LOAD16(p));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Store a 16-bit quantity to a byte array even if it's unaligned.
         * @param p The address to store to
         * @param v The value to store
         */
        inline void store16(void *p, uint16_t v) {
            TURBO_INTERNAL_UNALIGNED_STORE16(p, from_host16(v));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Load a 32-bit quantity from a byte array even if it's unaligned.
         * @param p The address to load from
         * @return The converted value
         */
        inline uint32_t load32(const void *p) {
            return to_host32(TURBO_INTERNAL_UNALIGNED_LOAD32(p));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Store a 32-bit quantity to a byte array even if it's unaligned.
         * @param p The address to store to
         * @param v The value to store
         */
        inline void store32(void *p, uint32_t v) {
            TURBO_INTERNAL_UNALIGNED_STORE32(p, from_host32(v));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Load a 64-bit quantity from a byte array even if it's unaligned.
         * @param p The address to load from
         * @return The converted value
         */
        inline uint64_t load64(const void *p) {
            return to_host64(TURBO_INTERNAL_UNALIGNED_LOAD64(p));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Store a 64-bit quantity to a byte array even if it's unaligned.
         * @param p The address to store to
         * @param v The value to store
         */
        inline void store64(void *p, uint64_t v) {
            TURBO_INTERNAL_UNALIGNED_STORE64(p, from_host64(v));
        }

    }  // namespace little_endian

    // Utilities to convert numbers between the current hosts's native byte
    // order and big-endian byte order (same as network byte order)
    //
    // Load/Store methods are alignment safe
    namespace big_endian {
#if TURBO_IS_LITTLE_ENDIAN

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t from_host16(uint16_t x) { return gbswap_16(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t to_host16(uint16_t x) { return gbswap_16(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t from_host32(uint32_t x) { return gbswap_32(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t to_host32(uint32_t x) { return gbswap_32(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t from_host64(uint64_t x) { return gbswap_64(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t to_host64(uint64_t x) { return gbswap_64(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Determine if the current host is little-endian.
         * @return True if the host is little-endian, false otherwise
         */
        inline constexpr bool is_little_endian() { return true; }

#elif TURBO_IS_BIG_ENDIAN

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t from_host16(uint16_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t to_host16(uint16_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t from_host32(uint32_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t to_host32(uint32_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t from_host64(uint64_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t to_host64(uint64_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Determine if the current host is little-endian.
         * @return True if the host is little-endian, false otherwise
         */
        inline constexpr bool is_little_endian() { return false; }

#endif /* ENDIAN */

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 8-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint8_t from_host(uint8_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t from_host(uint16_t x) { return from_host16(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t from_host(uint32_t x) { return from_host32(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t from_host(uint64_t x) { return from_host64(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 8-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint8_t to_host(uint8_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint16_t to_host(uint16_t x) { return to_host16(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint32_t to_host(uint32_t x) { return to_host32(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline uint64_t to_host(uint64_t x) { return to_host64(x); }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 8-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int8_t from_host(int8_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int16_t from_host(int16_t x) {
            return bit_cast<int16_t>(from_host16(bit_cast<uint16_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int32_t from_host(int32_t x) {
            return bit_cast<int32_t>(from_host32(bit_cast<uint32_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from host byte order to big-endian byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int64_t from_host(int64_t x) {
            return bit_cast<int64_t>(from_host64(bit_cast<uint64_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 8-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int8_t to_host(int8_t x) { return x; }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 16-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int16_t to_host(int16_t x) {
            return bit_cast<int16_t>(to_host16(bit_cast<uint16_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 32-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int32_t to_host(int32_t x) {
            return bit_cast<int32_t>(to_host32(bit_cast<uint32_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Convert a 64-bit quantity from big-endian byte order to host byte order.
         * @param x The value to convert
         * @return The converted value
         */
        inline int64_t to_host(int64_t x) {
            return bit_cast<int64_t>(to_host64(bit_cast<uint64_t>(x)));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Load a 16-bit quantity from a byte array even if it's unaligned.
         * @param p The address to load from
         * @return The converted value
         */
        inline uint16_t load16(const void *p) {
            return to_host16(TURBO_INTERNAL_UNALIGNED_LOAD16(p));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Store a 16-bit quantity to a byte array even if it's unaligned.
         * @param p The address to store to
         * @param v The value to store
         */
        inline void store16(void *p, uint16_t v) {
            TURBO_INTERNAL_UNALIGNED_STORE16(p, from_host16(v));
        }

        inline uint32_t load32(const void *p) {
            return to_host32(TURBO_INTERNAL_UNALIGNED_LOAD32(p));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Store a 32-bit quantity to a byte array even if it's unaligned.
         * @param p The address to store to
         * @param v The value to store
         */
        inline void store32(void *p, uint32_t v) {
            TURBO_INTERNAL_UNALIGNED_STORE32(p, from_host32(v));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Load a 64-bit quantity from a byte array even if it's unaligned.
         * @param p The address to load from
         * @return The converted value
         */
        inline uint64_t load64(const void *p) {
            return to_host64(TURBO_INTERNAL_UNALIGNED_LOAD64(p));
        }

        /**
         * @ingroup turbo_base_endian
         * @brief Store a 64-bit quantity to a byte array even if it's unaligned.
         * @param p The address to store to
         * @param v The value to store
         */
        inline void store64(void *p, uint64_t v) {
            TURBO_INTERNAL_UNALIGNED_STORE64(p, from_host64(v));
        }

    }  // namespace big_endian

}  // namespace turbo

#endif  // TURBO_BASE_INTERNAL_ENDIAN_H_
