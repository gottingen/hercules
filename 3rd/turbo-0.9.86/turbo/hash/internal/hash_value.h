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


#ifndef TURBO_HASH_INTERNAL_HASH_VALUE_H_
#define TURBO_HASH_INTERNAL_HASH_VALUE_H_

#include <type_traits>

namespace turbo::hash_internal {

    // -----------------------------------------------------------------------------
    // hash_value for Basic Types
    // -----------------------------------------------------------------------------

    // Note: Default `hash_value` implementations live in `hash_internal`. This
    // allows us to block lexical scope lookup when doing an unqualified call to
    // `hash_value` below. User-defined implementations of `hash_value` can
    // only be found via ADL.

    // hash_value() for hashing bool values
    //
    // We use SFINAE to ensure that this overload only accepts bool, not types that
    // are convertible to bool.
    template<typename H, typename B>
    typename std::enable_if<std::is_same<B, bool>::value, H>::type hash_value(
            H hash_state, B value) {
        return H::combine(std::move(hash_state),
                          static_cast<unsigned char>(value ? 1 : 0));
    }

    // hash_value() for hashing enum values
    template<typename H, typename Enum>
    typename std::enable_if<std::is_enum<Enum>::value, H>::type hash_value(
            H hash_state, Enum e) {
        // In practice, we could almost certainly just invoke hash_bytes directly,
        // but it's possible that a sanitizer might one day want to
        // store data in the unused bits of an enum. To avoid that risk, we
        // convert to the underlying type before hashing. Hopefully this will get
        // optimized away; if not, we can reopen discussion with c-toolchain-team.
        return H::combine(std::move(hash_state),
                          static_cast<typename std::underlying_type<Enum>::type>(e));
    }

    // hash_value() for hashing floating-point values
    template<typename H, typename Float>
    typename std::enable_if<std::is_same<Float, float>::value ||
                            std::is_same<Float, double>::value,
            H>::type
    hash_value(H hash_state, Float value) {
        return hash_internal::hash_bytes(std::move(hash_state),
                                         value == 0 ? 0 : value);
    }

    // Long double has the property that it might have extra unused bytes in it.
    // For example, in x86 sizeof(long double)==16 but it only really uses 80-bits
    // of it. This means we can't use hash_bytes on a long double and have to
    // convert it to something else first.
    template<typename H, typename LongDouble>
    typename std::enable_if<std::is_same<LongDouble, long double>::value, H>::type
    hash_value(H hash_state, LongDouble value) {
        const int category = std::fpclassify(value);
        switch (category) {
            case FP_INFINITE:
                // Add the sign bit to differentiate between +Inf and -Inf
                hash_state = H::combine(std::move(hash_state), std::signbit(value));
                break;

            case FP_NAN:
            case FP_ZERO:
            default:
                // Category is enough for these.
                break;

            case FP_NORMAL:
            case FP_SUBNORMAL:
                // We can't convert `value` directly to double because this would have
                // undefined behavior if the value is out of range.
                // std::frexp gives us a value in the range (-1, -.5] or [.5, 1) that is
                // guaranteed to be in range for `double`. The truncation is
                // implementation defined, but that works as long as it is deterministic.
                int exp;
                auto mantissa = static_cast<double>(std::frexp(value, &exp));
                hash_state = H::combine(std::move(hash_state), mantissa, exp);
        }

        return H::combine(std::move(hash_state), category);
    }

    // hash_value() for hashing pointers
    template<typename H, typename T>
    H hash_value(H hash_state, T *ptr) {
        auto v = reinterpret_cast<uintptr_t>(ptr);
        // Due to alignment, pointers tend to have low bits as zero, and the next few
        // bits follow a pattern since they are also multiples of some base value.
        // Mixing the pointer twice helps prevent stuck low bits for certain alignment
        // values.
        return H::combine(std::move(hash_state), v, v);
    }

    // hash_value() for hashing nullptr_t
    template<typename H>
    H hash_value(H hash_state, std::nullptr_t) {
        return H::combine(std::move(hash_state), static_cast<void *>(nullptr));
    }

    // hash_value() for hashing pointers-to-member
    template<typename H, typename T, typename C>
    H hash_value(H hash_state, T C::* ptr) {
        auto salient_ptm_size = [](std::size_t n) -> std::size_t {
#if defined(_MSC_VER)
            // Pointers-to-member-function on MSVC consist of one pointer plus 0, 1, 2,
                // or 3 ints. In 64-bit mode, they are 8-byte aligned and thus can contain
                // padding (namely when they have 1 or 3 ints). The value below is a lower
                // bound on the number of salient, non-padding bytes that we use for
                // hashing.
                if (alignof(T C::*) == alignof(int)) {
                  // No padding when all subobjects have the same size as the total
                  // alignment. This happens in 32-bit mode.
                  return n;
                } else {
                  // Padding for 1 int (size 16) or 3 ints (size 24).
                  // With 2 ints, the size is 16 with no padding, which we pessimize.
                  return n == 24 ? 20 : n == 16 ? 12 : n;
                }
#else
            // On other platforms, we assume that pointers-to-members do not have
            // padding.
#ifdef __cpp_lib_has_unique_object_representations
            static_assert(std::has_unique_object_representations<T C::*>::value);
#endif  // __cpp_lib_has_unique_object_representations
            return n;
#endif
        };
        return H::combine_contiguous(std::move(hash_state),
                                     reinterpret_cast<unsigned char *>(&ptr),
                                     salient_ptm_size(sizeof ptr));
    }

    // -----------------------------------------------------------------------------
    // hash_value for Composite Types
    // -----------------------------------------------------------------------------

    // hash_value() for hashing pairs
    template<typename H, typename T1, typename T2>
    typename std::enable_if<is_hashable<T1>::value && is_hashable<T2>::value,
            H>::type
    hash_value(H hash_state, const std::pair<T1, T2> &p) {
        return H::combine(std::move(hash_state), p.first, p.second);
    }

    // hash_tuple()
    //
    // Helper function for hashing a tuple. The third argument should
    // be an index_sequence running from 0 to tuple_size<Tuple> - 1.
    template<typename H, typename Tuple, size_t... Is>
    H hash_tuple(H hash_state, const Tuple &t, std::index_sequence<Is...>) {
        return H::combine(std::move(hash_state), std::get<Is>(t)...);
    }

// hash_value for hashing tuples
    template<typename H, typename... Ts>
#if defined(_MSC_VER)
    // This SFINAE gets MSVC confused under some conditions. Let's just disable it
        // for now.
        H
#else   // _MSC_VER
    typename std::enable_if<std::conjunction<is_hashable<Ts>...>::value, H>::type
#endif  // _MSC_VER
    hash_value(H hash_state, const std::tuple<Ts...> &t) {
        return hash_internal::hash_tuple(std::move(hash_state), t,
                                         std::make_index_sequence<sizeof...(Ts)>());
    }

    // -----------------------------------------------------------------------------
    // hash_value for Pointers
    // -----------------------------------------------------------------------------

    // hash_value for hashing unique_ptr
    template<typename H, typename T, typename D>
    H hash_value(H hash_state, const std::unique_ptr<T, D> &ptr) {
        return H::combine(std::move(hash_state), ptr.get());
    }

    // hash_value for hashing shared_ptr
    template<typename H, typename T>
    H hash_value(H hash_state, const std::shared_ptr<T> &ptr) {
        return H::combine(std::move(hash_state), ptr.get());
    }

    // -----------------------------------------------------------------------------
    // hash_value for String-Like Types
    // -----------------------------------------------------------------------------

    // hash_value for hashing strings
    //
    // All the string-like types supported here provide the same hash expansion for
    // the same character sequence. These types are:
    //
    //  - `turbo::Cord`
    //  - `std::string` (and std::basic_string<char, std::char_traits<char>, A> for
    //      any allocator A)
    //  - `std::string_view` and `std::string_view`
    //
    // For simplicity, we currently support only `char` strings. This support may
    // be broadened, if necessary, but with some caution - this overload would
    // misbehave in cases where the traits' `eq()` member isn't equivalent to `==`
    // on the underlying character type.
    template<typename H>
    H hash_value(H hash_state, std::string_view str) {
        return H::combine(
                H::combine_contiguous(std::move(hash_state), str.data(), str.size()),
                str.size());
    }

    // Support std::wstring, std::u16string and std::u32string.
    template<typename Char, typename Alloc, typename H,
            typename = std::enable_if_t<std::is_same<Char, wchar_t>::value ||
                                        std::is_same<Char, char16_t>::value ||
                                        std::is_same<Char, char32_t>::value>>
    H hash_value(
            H hash_state,
            const std::basic_string<Char, std::char_traits<Char>, Alloc> &str) {
        return H::combine(
                H::combine_contiguous(std::move(hash_state), str.data(), str.size()),
                str.size());
    }

    // -----------------------------------------------------------------------------
    // hash_value for Sequence Containers
    // -----------------------------------------------------------------------------

    // hash_value for hashing std::array
    template<typename H, typename T, size_t N>
    typename std::enable_if<is_hashable<T>::value, H>::type hash_value(
            H hash_state, const std::array<T, N> &array) {
        return H::combine_contiguous(std::move(hash_state), array.data(),
                                     array.size());
    }

    // hash_value for hashing std::deque
    template<typename H, typename T, typename Allocator>
    typename std::enable_if<is_hashable<T>::value, H>::type hash_value(
            H hash_state, const std::deque<T, Allocator> &deque) {
        // TODO(gromer): investigate a more efficient implementation taking
        // advantage of the chunk structure.
        for (const auto &t: deque) {
            hash_state = H::combine(std::move(hash_state), t);
        }
        return H::combine(std::move(hash_state), deque.size());
    }

    // hash_value for hashing std::forward_list
    template<typename H, typename T, typename Allocator>
    typename std::enable_if<is_hashable<T>::value, H>::type hash_value(
            H hash_state, const std::forward_list<T, Allocator> &list) {
        size_t size = 0;
        for (const T &t: list) {
            hash_state = H::combine(std::move(hash_state), t);
            ++size;
        }
        return H::combine(std::move(hash_state), size);
    }

    // hash_value for hashing std::list
    template<typename H, typename T, typename Allocator>
    typename std::enable_if<is_hashable<T>::value, H>::type hash_value(
            H hash_state, const std::list<T, Allocator> &list) {
        for (const auto &t: list) {
            hash_state = H::combine(std::move(hash_state), t);
        }
        return H::combine(std::move(hash_state), list.size());
    }

    // hash_value for hashing std::vector
    //
    // Do not use this for vector<bool> on platforms that have a working
    // implementation of std::hash. It does not have a .data(), and a fallback for
    // std::hash<> is most likely faster.
    template<typename H, typename T, typename Allocator>
    typename std::enable_if<is_hashable<T>::value && !std::is_same<T, bool>::value,
            H>::type
    hash_value(H hash_state, const std::vector<T, Allocator> &vector) {
        return H::combine(H::combine_contiguous(std::move(hash_state), vector.data(),
                                                vector.size()),
                          vector.size());
    }

// hash_value special cases for hashing std::vector<bool>

#if TURBO_IS_BIG_ENDIAN && \
    (defined(__GLIBCXX__) || defined(__GLIBCPP__))

    // std::hash in libstdc++ does not work correctly with vector<bool> on Big
        // Endian platforms therefore we need to implement a custom hash_value for
        // it. More details on the bug:
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102531
        template <typename H, typename T, typename Allocator>
        typename std::enable_if<is_hashable<T>::value && std::is_same<T, bool>::value,
                                H>::type
        hash_value(H hash_state, const std::vector<T, Allocator>& vector) {
          typename H::TurboInternalPiecewiseCombiner combiner;
          for (const auto& i : vector) {
            unsigned char c = static_cast<unsigned char>(i);
            hash_state = combiner.add_buffer(std::move(hash_state), &c, sizeof(c));
          }
          return H::combine(combiner.finalize(std::move(hash_state)), vector.size());
        }
#else

    // When not working around the libstdc++ bug above, we still have to contend
    // with the fact that std::hash<vector<bool>> is often poor quality, hashing
    // directly on the internal words and on no other state.  On these platforms,
    // vector<bool>{1, 1} and vector<bool>{1, 1, 0} hash to the same value.
    //
    // Mixing in the size (as we do in our other vector<> implementations) on top
    // of the library-provided hash implementation avoids this QOI issue.
    template<typename H, typename T, typename Allocator>
    typename std::enable_if<is_hashable<T>::value && std::is_same<T, bool>::value,
            H>::type
    hash_value(H hash_state, const std::vector<T, Allocator> &vector) {
        return H::combine(std::move(hash_state),
                          std::hash<std::vector<T, Allocator>>{}(vector),
                          vector.size());
    }

#endif

    // -----------------------------------------------------------------------------
    // hash_value for Ordered Associative Containers
    // -----------------------------------------------------------------------------

    // hash_value for hashing std::map
    template<typename H, typename Key, typename T, typename Compare,
            typename Allocator>
    typename std::enable_if<is_hashable<Key>::value && is_hashable<T>::value,
            H>::type
    hash_value(H hash_state, const std::map<Key, T, Compare, Allocator> &map) {
        for (const auto &t: map) {
            hash_state = H::combine(std::move(hash_state), t);
        }
        return H::combine(std::move(hash_state), map.size());
    }

    // hash_value for hashing std::multimap
    template<typename H, typename Key, typename T, typename Compare,
            typename Allocator>
    typename std::enable_if<is_hashable<Key>::value && is_hashable<T>::value,
            H>::type
    hash_value(H hash_state,
                   const std::multimap<Key, T, Compare, Allocator> &map) {
        for (const auto &t: map) {
            hash_state = H::combine(std::move(hash_state), t);
        }
        return H::combine(std::move(hash_state), map.size());
    }

    // hash_value for hashing std::set
    template<typename H, typename Key, typename Compare, typename Allocator>
    typename std::enable_if<is_hashable<Key>::value, H>::type hash_value(
            H hash_state, const std::set<Key, Compare, Allocator> &set) {
        for (const auto &t: set) {
            hash_state = H::combine(std::move(hash_state), t);
        }
        return H::combine(std::move(hash_state), set.size());
    }

    // hash_value for hashing std::multiset
    template<typename H, typename Key, typename Compare, typename Allocator>
    typename std::enable_if<is_hashable<Key>::value, H>::type hash_value(
            H hash_state, const std::multiset<Key, Compare, Allocator> &set) {
        for (const auto &t: set) {
            hash_state = H::combine(std::move(hash_state), t);
        }
        return H::combine(std::move(hash_state), set.size());
    }

    // -----------------------------------------------------------------------------
    // hash_value for Unordered Associative Containers
    // -----------------------------------------------------------------------------

    // hash_value for hashing std::unordered_set
    template<typename H, typename Key, typename Hash, typename KeyEqual,
            typename Alloc>
    typename std::enable_if<is_hashable<Key>::value, H>::type hash_value(
            H hash_state, const std::unordered_set<Key, Hash, KeyEqual, Alloc> &s) {
        return H::combine(
                H::combine_unordered(std::move(hash_state), s.begin(), s.end()),
                s.size());
    }

    // hash_value for hashing std::unordered_multiset
    template<typename H, typename Key, typename Hash, typename KeyEqual,
            typename Alloc>
    typename std::enable_if<is_hashable<Key>::value, H>::type hash_value(
            H hash_state,
            const std::unordered_multiset<Key, Hash, KeyEqual, Alloc> &s) {
        return H::combine(
                H::combine_unordered(std::move(hash_state), s.begin(), s.end()),
                s.size());
    }

    // hash_value for hashing std::unordered_set
    template<typename H, typename Key, typename T, typename Hash,
            typename KeyEqual, typename Alloc>
    typename std::enable_if<is_hashable<Key>::value && is_hashable<T>::value,
            H>::type
    hash_value(H hash_state,
                   const std::unordered_map<Key, T, Hash, KeyEqual, Alloc> &s) {
        return H::combine(
                H::combine_unordered(std::move(hash_state), s.begin(), s.end()),
                s.size());
    }

    // hash_value for hashing std::unordered_multiset
    template<typename H, typename Key, typename T, typename Hash,
            typename KeyEqual, typename Alloc>
    typename std::enable_if<is_hashable<Key>::value && is_hashable<T>::value,
            H>::type
    hash_value(H hash_state,
                   const std::unordered_multimap<Key, T, Hash, KeyEqual, Alloc> &s) {
        return H::combine(
                H::combine_unordered(std::move(hash_state), s.begin(), s.end()),
                s.size());
    }

    // -----------------------------------------------------------------------------
    // hash_value for Wrapper Types
    // -----------------------------------------------------------------------------

    // hash_value for hashing std::reference_wrapper
    template<typename H, typename T>
    typename std::enable_if<is_hashable<T>::value, H>::type hash_value(
            H hash_state, std::reference_wrapper<T> opt) {
        return H::combine(std::move(hash_state), opt.get());
    }

    // hash_value for hashing std::optional
    template<typename H, typename T>
    typename std::enable_if<is_hashable<T>::value, H>::type hash_value(
            H hash_state, const std::optional<T> &opt) {
        if (opt) hash_state = H::combine(std::move(hash_state), *opt);
        return H::combine(std::move(hash_state), opt.has_value());
    }

    // VariantVisitor
    template<typename H>
    struct VariantVisitor {
        H &&hash_state;

        template<typename T>
        H operator()(const T &t) const {
            return H::combine(std::move(hash_state), t);
        }
    };

    // hash_value for hashing std::variant
    template<typename H, typename... T>
    typename std::enable_if<std::conjunction<is_hashable<T>...>::value, H>::type
    hash_value(H hash_state, const std::variant<T...> &v) {
        if (!v.valueless_by_exception()) {
            hash_state = std::visit(VariantVisitor<H>{std::move(hash_state)}, v);
        }
        return H::combine(std::move(hash_state), v.index());
    }


    // -----------------------------------------------------------------------------
    // hash_value for Other Types
    // -----------------------------------------------------------------------------

    // hash_value for hashing std::bitset is not defined on Little Endian
    // platforms, for the same reason as for vector<bool> (see std::vector above):
    // It does not expose the raw bytes, and a fallback to std::hash<> is most
    // likely faster.

#if TURBO_IS_BIG_ENDIAN && \
    (defined(__GLIBCXX__) || defined(__GLIBCPP__))
    // hash_value for hashing std::bitset
        //
        // std::hash in libstdc++ does not work correctly with std::bitset on Big Endian
        // platforms therefore we need to implement a custom hash_value for it. More
        // details on the bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102531
        template <typename H, size_t N>
        H hash_value(H hash_state, const std::bitset<N>& set) {
          typename H::TurboInternalPiecewiseCombiner combiner;
          for (int i = 0; i < N; i++) {
            unsigned char c = static_cast<unsigned char>(set[i]);
            hash_state = combiner.add_buffer(std::move(hash_state), &c, sizeof(c));
          }
          return H::combine(combiner.finalize(std::move(hash_state)), N);
        }
#endif

    // -----------------------------------------------------------------------------

    // hash_range_or_bytes()
    //
    // Mixes all values in the range [data, data+size) into the hash state.
    // This overload accepts only uniquely-represented types, and hashes them by
    // hashing the entire range of bytes.
    template<typename H, typename T>
    typename std::enable_if<is_uniquely_represented<T>::value, H>::type
    hash_range_or_bytes(H hash_state, const T *data, size_t size) {
        const auto *bytes = reinterpret_cast<const unsigned char *>(data);
        return H::combine_contiguous(std::move(hash_state), bytes, sizeof(T) * size);
    }

    // hash_range_or_bytes()
    template<typename H, typename T>
    typename std::enable_if<!is_uniquely_represented<T>::value, H>::type
    hash_range_or_bytes(H hash_state, const T *data, size_t size) {
        for (const auto end = data + size; data < end; ++data) {
            hash_state = H::combine(std::move(hash_state), *data);
        }
        return hash_state;
    }

#if defined(TURBO_INTERNAL_LEGACY_HASH_NAMESPACE) && \
    TURBO_META_INTERNAL_STD_HASH_SFINAE_FRIENDLY_
#define TURBO_HASH_INTERNAL_SUPPORT_LEGACY_HASH_ 1
#else
#define TURBO_HASH_INTERNAL_SUPPORT_LEGACY_HASH_ 0
#endif

}  // namespace turbo::hash_internal

#endif  // TURBO_HASH_INTERNAL_HASH_VALUE_H_
