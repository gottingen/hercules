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

#ifndef TURBO_HASH_INTERNAL_PIECE_WISE_COMBINER_H_
#define TURBO_HASH_INTERNAL_PIECE_WISE_COMBINER_H_

#include <cstddef>
#include <utility>

namespace turbo::hash_internal {

    // Internal detail: Large buffers are hashed in smaller chunks.  This function
    // returns the size of these chunks.
    constexpr size_t PiecewiseChunkSize() { return 1024; }

    // PiecewiseCombiner
    //
    // PiecewiseCombiner is an internal-only helper class for hashing a piecewise
    // buffer of `char` or `unsigned char` as though it were contiguous.  This class
    // provides two methods:
    //
    //   H add_buffer(state, data, size)
    //   H finalize(state)
    //
    // `add_buffer` can be called zero or more times, followed by a single call to
    // `finalize`.  This will produce the same hash expansion as concatenating each
    // buffer piece into a single contiguous buffer, and passing this to
    // `H::combine_contiguous`.
    //
    //  Example usage:
    //    PiecewiseCombiner combiner;
    //    for (const auto& piece : pieces) {
    //      state = combiner.add_buffer(std::move(state), piece.data, piece.size);
    //    }
    //    return combiner.finalize(std::move(state));
    class PiecewiseCombiner {
    public:
        PiecewiseCombiner() : position_(0) {}

        PiecewiseCombiner(const PiecewiseCombiner &) = delete;

        PiecewiseCombiner &operator=(const PiecewiseCombiner &) = delete;

        // PiecewiseCombiner::add_buffer()
        //
        // Appends the given range of bytes to the sequence to be hashed, which may
        // modify the provided hash state.
        template<typename H>
        H add_buffer(H state, const unsigned char *data, size_t size);

        template<typename H>
        H add_buffer(H state, const char *data, size_t size) {
            return add_buffer(std::move(state),
                              reinterpret_cast<const unsigned char *>(data), size);
        }

        // PiecewiseCombiner::finalize()
        //
        // Finishes combining the hash sequence, which may may modify the provided
        // hash state.
        //
        // Once finalize() is called, add_buffer() may no longer be called. The
        // resulting hash state will be the same as if the pieces passed to
        // add_buffer() were concatenated into a single flat buffer, and then provided
        // to H::combine_contiguous().
        template<typename H>
        H finalize(H state);

    private:
        unsigned char buf_[PiecewiseChunkSize()];
        size_t position_;
    };

}  // namespace turbo::hash_internal
#endif  // TURBO_HASH_INTERNAL_PIECE_WISE_COMBINER_H_
