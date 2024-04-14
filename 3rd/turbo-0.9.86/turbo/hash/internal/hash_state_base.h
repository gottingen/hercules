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
//
// Created by jeff on 23-12-12.
//

#ifndef TURBO_HASH_INTERNAL_HASH_STATE_BASE_H_
#define TURBO_HASH_INTERNAL_HASH_STATE_BASE_H_

#include "turbo/hash/internal/piece_wise_combiner.h"


namespace turbo::hash_internal {

    // is_uniquely_represented
    //
    // `is_uniquely_represented<T>` is a trait class that indicates whether `T`
    // is uniquely represented.
    //
    // A type is "uniquely represented" if two equal values of that type are
    // guaranteed to have the same bytes in their underlying storage. In other
    // words, if `a == b`, then `memcmp(&a, &b, sizeof(T))` is guaranteed to be
    // zero. This property cannot be detected automatically, so this trait is false
    // by default, but can be specialized by types that wish to assert that they are
    // uniquely represented. This makes them eligible for certain optimizations.
    //
    // If you have any doubt whatsoever, do not specialize this template.
    // The default is completely safe, and merely disables some optimizations
    // that will not matter for most types. Specializing this template,
    // on the other hand, can be very hazardous.
    //
    // To be uniquely represented, a type must not have multiple ways of
    // representing the same value; for example, float and double are not
    // uniquely represented, because they have distinct representations for
    // +0 and -0. Furthermore, the type's byte representation must consist
    // solely of user-controlled data, with no padding bits and no compiler-
    // controlled data such as vptrs or sanitizer metadata. This is usually
    // very difficult to guarantee, because in most cases the compiler can
    // insert data and padding bits at its own discretion.
    //
    // If you specialize this template for a type `T`, you must do so in the file
    // that defines that type (or in this file). If you define that specialization
    // anywhere else, `is_uniquely_represented<T>` could have different meanings
    // in different places.
    //
    // The Enable parameter is meaningless; it is provided as a convenience,
    // to support certain SFINAE techniques when defining specializations.
    template<typename T, typename Enable = void>
    struct is_uniquely_represented : std::false_type {
    };

    // is_uniquely_represented<unsigned char>
    //
    // unsigned char is a synonym for "byte", so it is guaranteed to be
    // uniquely represented.
    template<>
    struct is_uniquely_represented<unsigned char> : std::true_type {
    };

    // is_uniquely_represented for non-standard integral types
    //
    // Integral types other than bool should be uniquely represented on any
    // platform that this will plausibly be ported to.
    template<typename Integral>
    struct is_uniquely_represented<
            Integral, typename std::enable_if<std::is_integral<Integral>::value>::type>
            : std::true_type {
    };

    // is_uniquely_represented<bool>
    //
    //
    template<>
    struct is_uniquely_represented<bool> : std::false_type {
    };

    // is_hashable()
    //
    // Trait class which returns true if T is hashable by the turbo::Hash framework.
    // Used for the hash_value implementations for composite types below.
    template<typename T>
    struct is_hashable;

    // HashStateBase
    //
    // An internal implementation detail that contains common implementation details
    // for all of the "hash state objects" objects generated by Turbo.  This is not
    // a public API; users should not create classes that inherit from this.
    //
    // A hash state object is the template argument `H` passed to `hash_value`.
    // It represents an intermediate state in the computation of an unspecified hash
    // algorithm. `HashStateBase` provides a CRTP style base class for hash state
    // implementations. Developers adding type support for `turbo::Hash` should not
    // rely on any parts of the state object other than the following member
    // functions:
    //
    //   * HashStateBase::combine()
    //   * HashStateBase::combine_contiguous()
    //   * HashStateBase::combine_unordered()
    //
    // A derived hash state class of type `H` must provide a public member function
    // with a signature similar to the following:
    //
    //    `static H combine_contiguous(H state, const unsigned char*, size_t)`.
    //
    // It must also provide a private template method named run_combine_unordered.
    //
    // A "consumer" is a 1-arg functor returning void.  Its argument is a reference
    // to an inner hash state object, and it may be called multiple times.  When
    // called, the functor consumes the entropy from the provided state object,
    // and resets that object to its empty state.
    //
    // A "combiner" is a stateless 2-arg functor returning void.  Its arguments are
    // an inner hash state object and an ElementStateConsumer functor.  A combiner
    // uses the provided inner hash state object to hash each element of the
    // container, passing the inner hash state object to the consumer after hashing
    // each element.
    //
    // Given these definitions, a derived hash state class of type H
    // must provide a private template method with a signature similar to the
    // following:
    //
    //    `template <typename CombinerT>`
    //    `static H run_combine_unordered(H outer_state, CombinerT combiner)`
    //
    // This function is responsible for constructing the inner state object and
    // providing a consumer to the combiner.  It uses side effects of the consumer
    // and combiner to mix the state of each element in an order-independent manner,
    // and uses this to return an updated value of `outer_state`.
    //
    // This inside-out approach generates efficient object code in the normal case,
    // but allows us to use stack storage to implement the turbo::HashState type
    // erasure mechanism (avoiding heap allocations while hashing).
    //
    // `HashStateBase` will provide a complete implementation for a hash state
    // object in terms of these two methods.
    //
    // Example:
    //
    //   // Use CRTP to define your derived class.
    //   struct MyHashState : HashStateBase<MyHashState> {
    //       static H combine_contiguous(H state, const unsigned char*, size_t);
    //       using MyHashState::HashStateBase::combine;
    //       using MyHashState::HashStateBase::combine_contiguous;
    //       using MyHashState::HashStateBase::combine_unordered;
    //     private:
    //       template <typename CombinerT>
    //       static H run_combine_unordered(H state, CombinerT combiner);
    //   };
    template<typename H>
    class HashStateBase {
    public:
        // HashStateBase::combine()
        //
        // Combines an arbitrary number of values into a hash state, returning the
        // updated state.
        //
        // Each of the value types `T` must be separately hashable by the Turbo
        // hashing framework.
        //
        // NOTE:
        //
        //   state = H::combine(std::move(state), value1, value2, value3);
        //
        // is guaranteed to produce the same hash expansion as:
        //
        //   state = H::combine(std::move(state), value1);
        //   state = H::combine(std::move(state), value2);
        //   state = H::combine(std::move(state), value3);
        template<typename T, typename... Ts>
        static H combine(H state, const T &value, const Ts &... values);

        static H combine(H state) { return state; }

        // HashStateBase::combine_contiguous()
        //
        // Combines a contiguous array of `size` elements into a hash state, returning
        // the updated state.
        //
        // NOTE:
        //
        //   state = H::combine_contiguous(std::move(state), data, size);
        //
        // is NOT guaranteed to produce the same hash expansion as a for-loop (it may
        // perform internal optimizations).  If you need this guarantee, use the
        // for-loop instead.
        template<typename T>
        static H combine_contiguous(H state, const T *data, size_t size);

        template<typename I>
        static H combine_unordered(H state, I begin, I end);

        using TurboInternalPiecewiseCombiner = PiecewiseCombiner;

        template<typename T>
        using is_hashable = turbo::hash_internal::is_hashable<T>;

    private:
        // Common implementation of the iteration step of a "combiner", as described
        // above.
        template<typename I>
        struct CombineUnorderedCallback {
            I begin;
            I end;

            template<typename InnerH, typename ElementStateConsumer>
            void operator()(InnerH inner_state, ElementStateConsumer cb) {
                for (; begin != end; ++begin) {
                    inner_state = H::combine(std::move(inner_state), *begin);
                    cb(inner_state);
                }
            }
        };
    };

    // hash_bytes()
    //
    // Convenience function that combines `hash_state` with the byte representation
    // of `value`.
    template<typename H, typename T>
    H hash_bytes(H hash_state, const T &value) {
        const unsigned char *start = reinterpret_cast<const unsigned char *>(&value);
        return H::combine_contiguous(std::move(hash_state), start, sizeof(value));
    }
}  // namespace turbo::hash_internal
#endif  // TURBO_HASH_INTERNAL_HASH_STATE_BASE_H_
