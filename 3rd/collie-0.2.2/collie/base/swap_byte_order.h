// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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

#ifndef COLLIE_BASE_SWAP_BYTE_ORDER_H_
#define COLLIE_BASE_SWAP_BYTE_ORDER_H_


#include <collie/base/bit.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__linux__) || defined(__GNU__) || defined(__HAIKU__) ||            \
    defined(__Fuchsia__) || defined(__EMSCRIPTEN__)
#include <endian.h>
#elif defined(_AIX)
#include <sys/machine.h>
#elif defined(__sun)
/* Solaris provides _BIG_ENDIAN/_LITTLE_ENDIAN selector in sys/types.h */
#include <sys/types.h>
#define BIG_ENDIAN 4321
#define LITTLE_ENDIAN 1234
#if defined(_BIG_ENDIAN)
#define BYTE_ORDER BIG_ENDIAN
#else
#define BYTE_ORDER LITTLE_ENDIAN
#endif
#elif defined(__MVS__)
#define BIG_ENDIAN 4321
#define LITTLE_ENDIAN 1234
#define BYTE_ORDER BIG_ENDIAN
#else
#if !defined(BYTE_ORDER) && !defined(_WIN32)
#include <machine/endian.h>
#endif
#endif

namespace collie {

    namespace sys {

#if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && BYTE_ORDER == BIG_ENDIAN
        constexpr bool IsBigEndianHost = true;
#define COLLIE_BIG_ENDIAN
#else
#define COLLIE_LITTLE_ENDIAN
        constexpr bool IsBigEndianHost = false;
#endif

        static const bool IsLittleEndianHost = !IsBigEndianHost;

        inline unsigned char      get_swapped_bytes(unsigned char      C) { return collie::byteswap(C); }
        inline   signed char      get_swapped_bytes( signed  char      C) { return collie::byteswap(C); }
        inline          char      get_swapped_bytes(         char      C) { return collie::byteswap(C); }

        inline unsigned short     get_swapped_bytes(unsigned short     C) { return collie::byteswap(C); }
        inline   signed short     get_swapped_bytes(  signed short     C) { return collie::byteswap(C); }

        inline unsigned int       get_swapped_bytes(unsigned int       C) { return collie::byteswap(C); }
        inline   signed int       get_swapped_bytes(  signed int       C) { return collie::byteswap(C); }

        inline unsigned long      get_swapped_bytes(unsigned long      C) { return collie::byteswap(C); }
        inline   signed long      get_swapped_bytes(  signed long      C) { return collie::byteswap(C); }

        inline unsigned long long get_swapped_bytes(unsigned long long C) { return collie::byteswap(C); }
        inline   signed long long get_swapped_bytes(  signed long long C) { return collie::byteswap(C); }

        inline float get_swapped_bytes(float C) {
            union {
                uint32_t i;
                float f;
            } in, out;
            in.f = C;
            out.i = collie::byteswap(in.i);
            return out.f;
        }

        inline double get_swapped_bytes(double C) {
            union {
                uint64_t i;
                double d;
            } in, out;
            in.d = C;
            out.i = collie::byteswap(in.i);
            return out.d;
        }

        template <typename T>
        inline std::enable_if_t<std::is_enum_v<T>, T> get_swapped_bytes(T C) {
            return static_cast<T>(
                    collie::byteswap(static_cast<std::underlying_type_t<T>>(C)));
        }

        template<typename T>
        inline void swap_byte_order(T &Value) {
            Value = get_swapped_bytes(Value);
        }

    } // end namespace sys
} // end namespace collie

#endif  // COLLIE_BASE_SWAP_BYTE_ORDER_H_
