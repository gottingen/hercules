// Copyright 2018 The Turbo Authors.
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
// Define the default Hash and Eq functions for SwissTable containers.
//
// std::hash<T> and std::equal_to<T> are not appropriate hash and equal
// functions for SwissTable containers. There are two reasons for this.
//
// SwissTable containers are power of 2 sized containers:
//
// This means they use the lower bits of the hash value to find the slot for
// each entry. The typical hash function for integral types is the identity.
// This is a very weak hash function for SwissTable and any power of 2 sized
// hashtable implementation which will lead to excessive collisions. For
// SwissTable we use murmur3 style mixing to reduce collisions to a minimum.
//
// SwissTable containers support heterogeneous lookup:
//
// In order to make heterogeneous lookup work, hash and equal functions must be
// polymorphic. At the same time they have to satisfy the same requirements the
// C++ standard imposes on hash functions and equality operators. That is:
//
//   if hash_default_eq<T>(a, b) returns true for any a and b of type T, then
//   hash_default_hash<T>(a) must equal hash_default_hash<T>(b)
//
// For SwissTable containers this requirement is relaxed to allow a and b of
// any and possibly different types. Note that like the standard the hash and
// equal functions are still bound to T. This is important because some type U
// can be hashed by/tested for equality differently depending on T. A notable
// example is `const char*`. `const char*` is treated as a c-style string when
// the hash function is hash<std::string> but as a pointer when the hash
// function is hash<void*>.
//
#ifndef TURBO_CONTAINER_INTERNAL_IGNORE_CASE_HASH_FUNCTION_H_
#define TURBO_CONTAINER_INTERNAL_IGNORE_CASE_HASH_FUNCTION_H_

#include <stdint.h>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <string_view>
#include "turbo/hash/hash.h"
#include "turbo/platform/port.h"
#include "turbo/strings/str_case_conv.h"
#include "turbo/strings/match.h"

namespace turbo::container_internal {

    template<typename  T>
    struct IgnoreCaseHashEq;

    struct IgnoreCaseStringHash {
        using is_transparent = void;

        size_t operator()(std::string_view v) const {
            auto str = str_to_lower(v);
            return turbo::Hash<std::string_view>{}(str);
        }
    };

    struct IgnoreCaseStringEq {
        using is_transparent = void;

        bool operator()(std::string_view lhs, std::string_view rhs) const {
            return str_equals_ignore_case(lhs, rhs);
        }
    };

    // Supports heterogeneous lookup for string-like elements.
    struct IgnoreCaseStringHashEq {
        using Hash = IgnoreCaseStringHash;
        using Eq = IgnoreCaseStringEq;
    };

    template<>
    struct IgnoreCaseHashEq<std::string> : IgnoreCaseStringHashEq {
    };
    template<>
    struct IgnoreCaseHashEq<std::string_view> : IgnoreCaseStringHashEq {
    };

    template<class T>
    using hash_default_ignore_case_hash = typename container_internal::IgnoreCaseHashEq<T>::Hash;

    // This header's visibility is restricted.  If you need to access the default
    // key equal please use the container's ::key_equal alias instead.
    //
    // Example: typename Eq = typename turbo::flat_hash_map<K, V, Hash>::key_equal
    template<class T>
    using hash_default_ignore_case_eq = typename container_internal::IgnoreCaseHashEq<T>::Eq;

}  // namespace turbo::container_internal

#endif  // TURBO_CONTAINER_INTERNAL_IGNORE_CASE_HASH_FUNCTION_H_
