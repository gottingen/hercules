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


#ifndef TURBO_CONTAINER_BATCH_H_
#define TURBO_CONTAINER_BATCH_H_

#include "turbo/format/print.h"
#include "turbo/meta/type_traits.h"
#include <initializer_list>
#include <vector>

namespace turbo {

    template<typename T, size_t N>
    class Batch {
    public:
        static const size_t WIDTH = N;
        using value_type = T;

        Batch() {
            _data.fill(T{});
        }

        template<typename S>
        Batch(const S &initial_value) {
            _data.fill(static_cast<T>(initial_value));
        }

        template<typename S>
        Batch(const std::initializer_list<S> &list) {
            TURBO_ASSERT(list.size() == N);
            size_t i = 0;
            for (auto &item : list) {
                _data[i++] = static_cast<T>(item);
            }
        }

        template<typename S>
        Batch(const std::vector<S> &list) {
            TURBO_ASSERT(list.size() == N);
            size_t i = 0;
            for (auto &item : list) {
                _data[i++] = static_cast<T>(item);
            }
        }

        template<typename S>
        Batch(const Batch<S, N> &other) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] = static_cast<T>(other[i]);
            }
        }

        template<typename S>
        Batch(const Batch<S, N> &&other) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] = static_cast<T>(other[i]);
            }
        }

        template<typename S>
        Batch &operator=(const Batch<S, N> &other) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] = static_cast<T>(other[i]);
            }
            return *this;
        }

        template<typename S>
        Batch &operator=(const Batch<S, N> &&other) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] = static_cast<T>(other[i]);
            }
            return *this;
        }

        template<typename S>
        Batch &operator=(const std::initializer_list<S> &list) {
            TURBO_ASSERT(list.size() == N);
            size_t i = 0;
            for (auto &item : list) {
                _data[i++] = static_cast<T>(item);
            }
            return *this;
        }

        template<typename S>
        Batch &operator=(const std::vector<S> &list) {
            TURBO_ASSERT(list.size() == N);
            size_t i = 0;
            for (auto &item : list) {
                _data[i++] = static_cast<T>(item);
            }
            return *this;
        }

        template<typename S>
        Batch &operator=(const S &scalar) {
            _data.fill(static_cast<T>(scalar));
            return *this;
        }

        template<typename S>
        Batch &operator+=(const Batch<S, N> &rhs) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] += rhs[i];
            }
            return *this;
        }

        template<typename S>
        Batch &operator+=(const S &scalar) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] += scalar;
            }
            return *this;
        }

        template<typename S>
        Batch &operator-=(const Batch<S, N> &rhs) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] -= rhs[i];
            }
            return *this;
        }

        template<typename S>
        Batch &operator-=(const S &scalar) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] -= scalar;
            }
            return *this;
        }

        template<typename S>
        Batch &operator*=(const Batch<S, N> &rhs) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] *= rhs[i];
            }
            return *this;
        }

        template<typename S>
        Batch &operator*=(const S &scalar) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] *= scalar;
            }
            return *this;
        }

        template<typename S>
        Batch &operator/=(const Batch<S, N> &rhs) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] /= rhs[i];
            }
            return *this;
        }

        template<typename S>
        Batch &operator/=(const S &scalar) {
            for (size_t i = 0; i < N; ++i) {
                _data[i] /= scalar;
            }
            return *this;
        }

        template<typename S>
        Batch operator+(const Batch<S, N> &rhs) const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = _data[i] + rhs[i];
            }
            return result;
        }

        template<typename S>
        Batch operator+(const S &scalar) const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = _data[i] + scalar;
            }
            return result;
        }

        template<typename S>
        Batch operator-(const Batch<S, N> &rhs) const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = _data[i] - rhs[i];
            }
            return result;
        }

        template<typename S>
        Batch operator-(const S &scalar) const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = _data[i] - scalar;
            }
            return result;
        }

        template<typename S>
        Batch operator*(const Batch<S, N> &rhs) const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = _data[i] * rhs[i];
            }
            return result;
        }

        template<typename S>
        Batch operator*(const S &scalar) const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = _data[i] * scalar;
            }
            return result;
        }

        template<typename S>
        Batch operator/(const Batch<S, N> &rhs) const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = _data[i] / rhs[i];
            }
            return result;
        }

        template<typename S>
        Batch operator/(const S &scalar) const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = _data[i] / scalar;
            }
            return result;
        }

        Batch operator-() const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = -_data[i];
            }
            return result;
        }

        Batch operator+() const {
            Batch result;
            for (size_t i = 0; i < N; ++i) {
                result[i] = +_data[i];
            }
            return result;
        }

        bool operator==(const Batch &rhs) const {
            for (size_t i = 0; i < N; ++i) {
                if (!(_data[i] == rhs._data[i])) {
                    return false;
                }
            }
            return true;
        }

        bool operator<(const Batch &rhs) const {
            for (size_t i = 0; i < N; ++i) {
                if (!(_data[i] < rhs._data[i])) {
                    return false;
                }
            }
            return true;
        }

        bool operator>(const Batch &rhs) const {
            for (size_t i = 0; i < N; ++i) {
                if (!(_data[i] > rhs._data[i])) {
                    return false;
                }
            }
            return true;
        }

        bool operator<=(const Batch &rhs) const {
            for (size_t i = 0; i < N; ++i) {
                if (!(_data[i] <= rhs._data[i])) {
                    return false;
                }
            }
            return true;
        }

        bool operator>=(const Batch &rhs) const {
            for (size_t i = 0; i < N; ++i) {
                if (!(_data[i] >= rhs._data[i])) {
                    return false;
                }
            }
            return true;
        }

        bool operator!=(const Batch &rhs) const {
            return !operator==(rhs);
        }

        T &operator[](size_t index) {
            TURBO_ASSERT(index < N);
            return _data[index]; }

        const T &operator[](size_t index) const {
            TURBO_ASSERT(index < N);
            return _data[index]; }

        [[nodiscard]] size_t size() const { return N; }

    private:
        std::array<T, N> _data;
    };

    template <typename T, size_t N, typename Char>
    struct formatter<Batch<T, N>, Char> : public formatter<T, Char> {
        constexpr auto parse(basic_format_parse_context<Char> &ctx)
        -> decltype(ctx.begin()) {
            return formatter<T, Char>::parse(ctx);
        }

        auto format(const Batch<T, N> &batch, format_context &ctx) -> decltype(ctx.out()) {
            auto out = ctx.out();
            out = format_to(out, "[");
            ctx.advance_to(out);
            for(size_t i = 0; i < N; ++i) {
                if(i != 0) {
                    out = format_to(out, ", ");
                    ctx.advance_to(out);
                }
                out = formatter<T, Char>::format(batch[i],ctx);
                ctx.advance_to(out);
            }
            out = format_to(out, "]");
            ctx.advance_to(out);
            return out;
        }

    };

    template<typename T, size_t N>
    std::ostream &operator<<(std::ostream &os, const Batch<T, N> &vec) {
        os << '[';
        if (N != 0) {
            os << vec[0];
            for (size_t i = 1; i < N; ++i) {
                os << ',' << vec[i];
            }
        }
        os << ']';
        return os;
    }

    template<typename T>
    struct is_batch : public std::false_type {
    };

    template<typename T, size_t N>
    struct is_batch<Batch<T, N> > : public std::true_type {
    };


}  // namespace turbo

#endif // TURBO_CONTAINER_BATCH_H_
