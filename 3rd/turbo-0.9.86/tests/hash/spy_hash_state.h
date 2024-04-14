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

#ifndef TURBO_HASH_INTERNAL_SPY_HASH_STATE_H_
#define TURBO_HASH_INTERNAL_SPY_HASH_STATE_H_

#include <algorithm>
#include <ostream>
#include <string>
#include <vector>

#include "turbo/hash/hash.h"
#include "turbo/strings/match.h"
#include "turbo/format/format.h"

namespace turbo::hash_internal {


    // SpyHashState is an implementation of the HashState API that simply
    // accumulates all input bytes in an internal buffer. This makes it useful
    // for testing hash_value overloads (so long as they are templated on the
    // HashState parameter), since it can report the exact hash representation
    // that the hash_value overload produces.
    //
    // Sample usage:
    // EXPECT_EQ(SpyHashState::combine(SpyHashState(), foo),
    //           SpyHashState::combine(SpyHashState(), bar));
    template<typename T>
    class SpyHashStateImpl : public HashStateBase<SpyHashStateImpl<T>> {
    public:
        SpyHashStateImpl() : error_(std::make_shared<std::optional<std::string>>()) {
            static_assert(std::is_void<T>::value, "");
        }

        // Move-only
        SpyHashStateImpl(const SpyHashStateImpl &) = delete;

        SpyHashStateImpl &operator=(const SpyHashStateImpl &) = delete;

        SpyHashStateImpl(SpyHashStateImpl &&other) noexcept {
            *this = std::move(other);
        }

        SpyHashStateImpl &operator=(SpyHashStateImpl &&other) noexcept {
            hash_representation_ = std::move(other.hash_representation_);
            error_ = other.error_;
            moved_from_ = other.moved_from_;
            other.moved_from_ = true;
            return *this;
        }

        template<typename U>
        SpyHashStateImpl(SpyHashStateImpl<U> &&other) {  // NOLINT
            hash_representation_ = std::move(other.hash_representation_);
            error_ = other.error_;
            moved_from_ = other.moved_from_;
            other.moved_from_ = true;
        }

        template<typename A, typename... Args>
        static SpyHashStateImpl combine(SpyHashStateImpl s, const A &a,
                                        const Args &... args) {
            // Pass an instance of SpyHashStateImpl<A> when trying to combine `A`. This
            // allows us to test that the user only uses this instance for combine calls
            // and does not call hash_value directly.
            // See hash_value implementation at the bottom.
            s = SpyHashStateImpl<A>::HashStateBase::combine(std::move(s), a);
            return SpyHashStateImpl::combine(std::move(s), args...);
        }

        static SpyHashStateImpl combine(SpyHashStateImpl s) {
            if (direct_turbo_hash_value_error_) {
                *s.error_ = "hash_value should not be invoked directly.";
            } else if (s.moved_from_) {
                *s.error_ = "Used moved-from instance of the hash state object.";
            }
            return s;
        }

        static void SetDirectTurboHashValueError() {
            direct_turbo_hash_value_error_ = true;
        }

        // Two SpyHashStateImpl objects are equal if they hold equal hash
        // representations.
        friend bool operator==(const SpyHashStateImpl &lhs,
                               const SpyHashStateImpl &rhs) {
            return lhs.hash_representation_ == rhs.hash_representation_;
        }

        friend bool operator!=(const SpyHashStateImpl &lhs,
                               const SpyHashStateImpl &rhs) {
            return !(lhs == rhs);
        }

        enum class CompareResult {
            kEqual,
            kASuffixB,
            kBSuffixA,
            kUnequal,
        };

        static CompareResult Compare(const SpyHashStateImpl &a,
                                     const SpyHashStateImpl &b) {
            const std::string a_flat = turbo::format_range("{}", a.hash_representation_, "");
            const std::string b_flat = turbo::format_range("{}", b.hash_representation_, "");
            if (a_flat == b_flat) return CompareResult::kEqual;
            if (turbo::ends_with(a_flat, b_flat)) return CompareResult::kBSuffixA;
            if (turbo::ends_with(b_flat, a_flat)) return CompareResult::kASuffixB;
            return CompareResult::kUnequal;
        }

        // operator<< prints the hash representation as a hex and ASCII dump, to
        // facilitate debugging.
        friend std::ostream &operator<<(std::ostream &out,
                                        const SpyHashStateImpl &hash_state) {
            out << "[\n";
            for (auto &s: hash_state.hash_representation_) {
                size_t offset = 0;
                for (char c: s) {
                    if (offset % 16 == 0) {
                        out << turbo::format("\n{:#04x}: ", offset);
                    }
                    if (offset % 2 == 0) {
                        out << " ";
                    }
                    out << turbo::format("{:02x}", c);
                    ++offset;
                }
                out << "\n";
            }
            return out << "]";
        }

        // The base case of the combine recursion, which writes raw bytes into the
        // internal buffer.
        static SpyHashStateImpl combine_contiguous(SpyHashStateImpl hash_state,
                                                   const unsigned char *begin,
                                                   size_t size) {
            const size_t large_chunk_stride = PiecewiseChunkSize();
            if (size > large_chunk_stride) {
                // Combining a large contiguous buffer must have the same effect as
                // doing it piecewise by the stride length, followed by the (possibly
                // empty) remainder.
                while (size >= large_chunk_stride) {
                    hash_state = SpyHashStateImpl::combine_contiguous(
                            std::move(hash_state), begin, large_chunk_stride);
                    begin += large_chunk_stride;
                    size -= large_chunk_stride;
                }
            }

            hash_state.hash_representation_.emplace_back(
                    reinterpret_cast<const char *>(begin), size);
            return hash_state;
        }

        using SpyHashStateImpl::HashStateBase::combine_contiguous;

        template<typename CombinerT>
        static SpyHashStateImpl run_combine_unordered(SpyHashStateImpl state,
                                                    CombinerT combiner) {
            UnorderedCombinerCallback cb;

            combiner(SpyHashStateImpl<void>{}, std::ref(cb));

            std::sort(cb.element_hash_representations.begin(),
                      cb.element_hash_representations.end());
            state.hash_representation_.insert(state.hash_representation_.end(),
                                              cb.element_hash_representations.begin(),
                                              cb.element_hash_representations.end());
            if (cb.error && cb.error->has_value()) {
                state.error_ = std::move(cb.error);
            }
            return state;
        }

        std::optional<std::string> error() const {
            if (moved_from_) {
                return "Returned a moved-from instance of the hash state object.";
            }
            return *error_;
        }

    private:
        template<typename U>
        friend
        class SpyHashStateImpl;

        struct UnorderedCombinerCallback {
            std::vector<std::string> element_hash_representations;
            std::shared_ptr<std::optional<std::string>> error;

            // The inner spy can have a different type.
            template<typename U>
            void operator()(SpyHashStateImpl<U> &inner) {
                element_hash_representations.push_back(
                        turbo::format_range("{}", inner.hash_representation_, ""));
                if (inner.error_->has_value()) {
                    error = std::move(inner.error_);
                }
                inner = SpyHashStateImpl<void>{};
            }
        };

        // This is true if SpyHashStateImpl<T> has been passed to a call of
        // hash_value with the wrong type. This detects that the user called
        // hash_value directly (because the hash state type does not match).
        static bool direct_turbo_hash_value_error_;

        std::vector<std::string> hash_representation_;
        // This is a shared_ptr because we want all instances of the particular
        // SpyHashState run to share the field. This way we can set the error for
        // use-after-move and all the copies will see it.
        std::shared_ptr<std::optional<std::string>> error_;
        bool moved_from_ = false;
    };

    template<typename T>
    bool SpyHashStateImpl<T>::direct_turbo_hash_value_error_;

    template<bool &B>
    struct OdrUse {
        constexpr OdrUse() {}

        bool &b = B;
    };

    template<void (*)()>
    struct RunOnStartup {
        static bool run;
        static constexpr OdrUse<run> kOdrUse{};
    };

    template<void (*f)()>
    bool RunOnStartup<f>::run = (f(), true);

    template<
            typename T, typename U,
            // Only trigger for when (T != U),
            typename = std::enable_if_t<!std::is_same<T, U>::value>,
            // This statement works in two ways:
            //  - First, it instantiates RunOnStartup and forces the initialization of
            //    `run`, which set the global variable.
            //  - Second, it triggers a SFINAE error disabling the overload to prevent
            //    compile time errors. If we didn't disable the overload we would get
            //    ambiguous overload errors, which we don't want.
            int = RunOnStartup<SpyHashStateImpl<T>::SetDirectTurboHashValueError>::run>
    void hash_value(SpyHashStateImpl<T>, const U &);

    using SpyHashState = SpyHashStateImpl<void>;

}  // namespace turbo::hash_internal

#endif  // TURBO_HASH_INTERNAL_SPY_HASH_STATE_H_
