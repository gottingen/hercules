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
#include "turbo/strings/unicode.h"

namespace turbo {
#define TURBO_UNICODE_ENABLE_INVARIANT_CHECK
#ifdef TURBO_UNICODE_ENABLE_INVARIANT_CHECK
    namespace {
        struct Invariant {
            Invariant &operator=(const Invariant &) = delete;

            explicit Invariant(const Unicode &s) noexcept: s_(s) {
                assert(s_.isSane());
            }

            ~Invariant() noexcept {
                assert(s_.isSane());
            }

        private:
            const Unicode &s_;
        };
    }  // namespace

#define UNICODE_INVARIANT_CHECK(s) Invariant invariant_checker(s)
#else
#define UNICODE_INVARIANT_CHECK(s)
#endif  // TURBO_UNICODE_ENABLE_INVARIANT_CHECK

    bool Unicode::isSane() const noexcept {
        return begin() <= end() && empty() == (size() == 0) && empty() == (begin() == end()) &&
               size() <= max_size() && capacity() <= max_size() && size() <= capacity() &&
               begin()[size()] == '\0';
    }

    const typename Unicode::size_type Unicode::npos;

    const Unicode::value_type *Unicode::data() const noexcept {
        return data_.data();
    }

    Unicode::operator self_view() const noexcept {
        return self_view(data_.data(), data_.size());
    }

    Unicode::self_view Unicode::view() const noexcept {
        return self_view(data_.data(), data_.size());
    }


    // default constructors
    Unicode &Unicode::operator=(const Unicode &other) {
        UNICODE_INVARIANT_CHECK(*this);
        if (TURBO_UNLIKELY(&other == this)) {
            return *this;
        }
        if (other.data_.category() == ContainerType::Category::isLarge) {
            ContainerType(other.data_).swap(data_);
            return *this;
        }
        return assign(other.data_.data(), other.data_.size());
    }

    Unicode &Unicode::operator=(Unicode &&other) noexcept {
        if (TURBO_UNLIKELY(&other == this)) {
            // Compatibility with std::basic_string<>,
            // C++11 21.4.2 [string.cons] / 23 requires self-move-assignment support.
            return *this;
        }
        // No need of this anymore
        this->~Unicode();
        // Move the goner into this
        new(&data_) ContainerType(std::move(other.data_));
        return *this;
    }

    // Member functions
    int Unicode::compare(const Unicode &other) const noexcept {
        return view().compare(other);
    }

    const Unicode::value_type *Unicode::c_str() const noexcept {
        return data_.c_str();
    }

    size_t Unicode::size() const noexcept {
        return view().size();
    }

    size_t Unicode::length() const noexcept {
        return view().size();
    }

    bool Unicode::empty() const noexcept {
        return size() == 0;
    }

    const Unicode::value_type &Unicode::front() const noexcept {
        return view().front();
    }

    const Unicode::value_type &Unicode::back() const noexcept {
        return view().back();
    }

    Unicode::value_type &Unicode::front() {
        return *(data_.mutableData());
    }

    Unicode::value_type &Unicode::back() {
        return *(data_.mutableData() + data_.size() - 1);
    }

    void Unicode::pop_back() {
        UNICODE_INVARIANT_CHECK(*this);
        assert(!empty());
        data_.shrink(1);
    }

    void Unicode::resizeNoInit(size_type n) {
        UNICODE_INVARIANT_CHECK(*this);
        auto size = this->size();
        if (n <= size) {
            data_.shrink(size - n);
        } else {
            auto const delta = n - size;
            data_.expandNoinit(delta);
        }
        assert(this->size() == n);
    }

    void Unicode::resize(size_type n, value_type c) {
        UNICODE_INVARIANT_CHECK(*this);
        auto size = this->size();
        if (n <= size) {
            data_.shrink(size - n);
        } else {
            auto const delta = n - size;
            auto pData = data_.expandNoinit(delta);
            strings_internal::podFill(pData, pData + delta, c);
        }
        assert(this->size() == n);
    }

    int64_t Unicode::capacity() const noexcept {
        return data_.capacity();
    }

    void Unicode::reserve(size_type res_arg) {
        data_.reserve(res_arg);
    }

    void Unicode::shrink_to_fit() {
        // Shrink only if slack memory is sufficiently large
        if (capacity() < size() * 3 / 2) {
            return;
        }
        ContainerType(data(), size()).swap(data_);
    }

    void Unicode::clear() {
        resize(0);
    }

    const Unicode::value_type &Unicode::operator[](size_type pos) const noexcept {
        return view().operator[](pos);
    }

    Unicode::value_type &Unicode::operator[](size_type pos) {
        return *(data_.mutableData() + pos);
    }

    const Unicode::value_type &Unicode::at(size_type n) const {
        if (n >= size()) {
            throw std::out_of_range("Unicode: index out of range");
        }
        return *(data_.data() + n);
    }

    Unicode::value_type &Unicode::at(size_type n) {
        if (n >= size()) {
            throw std::out_of_range("Unicode: index out of range");
        }
        return *(data_.mutableData() + n);
    }

    Unicode &Unicode::operator+=(self_view s) {
        return append(s);
    }

    Unicode &Unicode::operator+=(value_type c) {
        push_back(c);
        return *this;
    }

    Unicode &Unicode::operator+=(std::initializer_list<value_type> il) {
        append(il);
        return *this;
    }

    Unicode &Unicode::append(self_view str) {
        UNICODE_INVARIANT_CHECK(*this);
        auto s = str.data();
        auto n = str.size();

        if (TURBO_UNLIKELY(!n)) {
            // Unlikely but must be done
            return *this;
        }
        auto const oldSize = size();
        auto const oldData = data();
        auto pData = data_.expandNoinit(n, /* expGrowth = */ true);

        // Check for aliasing (rare). We could use "<=" here but in theory
        // those do not work for pointers unless the pointers point to
        // elements in the same array. For that reason we use
        // std::less_equal, which is guaranteed to offer a total order
        // over pointers. See discussion at http://goo.gl/Cy2ya for more
        // info.
        std::less_equal<const value_type *> le;
        if (TURBO_UNLIKELY(le(oldData, s) && !le(oldData + oldSize, s))) {
            assert(le(s + n, oldData + oldSize));
            // expandNoinit() could have moved the storage, restore the source.
            s = data() + (s - oldData);
            strings_internal::podMove(s, s + n, pData);
        } else {
            strings_internal::podCopy(s, s + n, pData);
        }

        assert(size() == oldSize + n);
        return *this;
    }

    Unicode &Unicode::append(self_view str, size_type pos, size_type n) {
        return append(str.substr(pos, n));
    }

    Unicode &Unicode::append(size_type n, value_type c) {
        UNICODE_INVARIANT_CHECK(*this);
        auto pData = data_.expandNoinit(n, /* expGrowth = */ true);
        strings_internal::podFill(pData, pData + n, c);
        return *this;
    }

    void Unicode::push_back(value_type c) {
        UNICODE_INVARIANT_CHECK(*this);
        data_.push_back(c);
    }

    Unicode &Unicode::assign(const Unicode &str) {
        if (TURBO_UNLIKELY(&str == this)) {
            return *this;
        }
        if (str.data_.category() == ContainerType::Category::isLarge) {
            ContainerType(str.data_).swap(data_);
            return *this;
        }
        return assign(str.data_.data(), str.data_.size());
    }

    Unicode &Unicode::assign(Unicode &&str) {
        return *this = std::move(str);
    }

    Unicode &Unicode::assign(const Unicode &str, size_type pos, size_type n) {
        auto sub_view = str.view().substr(pos, n);
        assign(sub_view.data(), sub_view.size());
        return *this;
    }

    Unicode &Unicode::assign(const value_type *s, size_type n) {
        UNICODE_INVARIANT_CHECK(*this);

        auto self_len = data_.size();
        if (n == 0) {
            data_.shrink(self_len);
        } else if (self_len >= n) {
            // s can alias this, we need to use podMove.
            strings_internal::podMove(s, s + n, data_.mutableData());
            data_.shrink(self_len - n);
            assert(size() == n);
        } else {
            // If n is larger than size(), s cannot alias this string's
            // storage.
            if (data_.isShared()) {
                ContainerType(s, n).swap(data_);
            } else {
                // Do not use exponential growth here: assign() should be tight,
                // to mirror the behavior of the equivalent constructor.
                data_.expandNoinit(n - self_len);
                strings_internal::podCopy(s, s + n, data_.data());
            }
        }

        assert(size() == n);
        return *this;
    }

    Unicode &Unicode::assign(const value_type *s) {
        return s ? assign(s, std::char_traits<value_type>::length(s)) : *this;
    }

    Unicode &Unicode::assign(self_view s) {
        if (s.category() == static_cast<int>(ContainerType::Category::isLarge)) {
            ContainerType(s.data(), s.size(), s.category()).swap(data_);
            return *this;
        }
        return assign(s.data(), s.size());
    }

    Unicode &Unicode::insert(size_type pos1, const Unicode &str) {
        return insert(pos1, str.view());
    }

    Unicode &Unicode::insert(size_type pos1, const Unicode &str, size_type pos2, size_type n) {
        return insert(pos1, str.view().substr(pos2, n));
    }

    Unicode &Unicode::insert(size_type pos, self_view s) {
        UNICODE_INVARIANT_CHECK(*this);
        auto n = s.size();
        auto oldSize = size();
        data_.expandNoinit(n, /* expGrowth = */ true);
        auto b = begin();
        strings_internal::podMove(b + pos, b + oldSize, b + pos + n);
        std::copy(s.begin(), s.end(), b + pos);
        return *this;
    }

    Unicode &Unicode::insert(size_type pos, const value_type *s) {
        return insert(pos, self_view(s));
    }

    Unicode &Unicode::insert(size_type pos, const value_type *s, size_type n) {
        return insert(pos, self_view(s, n));
    }

    Unicode &Unicode::insert(size_type pos, size_type n, value_type c) {
        UNICODE_INVARIANT_CHECK(*this);
        auto oldSize = size();
        data_.expandNoinit(n, /* expGrowth = */ true);
        auto b = begin();
        strings_internal::podMove(b + pos, b + oldSize, b + pos + n);
        strings_internal::podFill(b + pos, b + pos + n, c);
        return *this;
    }


    Unicode::operator std::basic_string_view<Unicode::value_type,
            std::char_traits<Unicode::value_type>>() const noexcept {
        return {data(), size()};
    }


    Unicode &Unicode::erase(size_type pos, size_type n) {
        UNICODE_INVARIANT_CHECK(*this);
        if (pos > size()) {
            throw std::out_of_range("Unicode: index out of range");
        }
        n = std::min(n, size_type(size()) - pos);
        std::copy(begin() + pos + n, end(), begin() + pos);
        resize(size() - n);
        return *this;
    }

    Unicode &Unicode::replace(size_type pos1, size_type n1, self_view s, size_type pos2, size_type n2) {
        return replace(pos1, n1, s.substr(pos2, n2));;
    }

    Unicode &Unicode::replace(size_type pos, size_type n1, self_view s) {
        UNICODE_INVARIANT_CHECK(*this);
        size_t len = size();
        if (pos > len) {
            throw std::out_of_range("Unicode: index out of range");
        }
        n1 = std::min(n1, len - pos);
        Unicode temp;
        temp.reserve(len - n1 + s.size());
        temp.append(substr(0, pos)).append(s).append(substr(pos + n1));
        *this = std::move(temp);
        return *this;
    }

    Unicode &Unicode::replace(size_type pos, size_type n1, self_view s, size_type n2) {
        return replace(pos, n1, s.substr(0, n2));
    }

    Unicode &Unicode::replace(size_type pos, size_type n1, size_type n2, value_type c) {
        UNICODE_INVARIANT_CHECK(*this);
        Unicode temp;
        temp.reserve(size() - n1 + size());
        temp.append(substr(0, pos)).append(n2, c).append(substr(pos + n1));
        *this = std::move(temp);
        return *this;
    }

    Unicode::size_type Unicode::find(self_view str, size_type pos) const {
        return view().find(str, pos);
    }

    Unicode::size_type Unicode::find(value_type c, size_type pos) const {
        return view().find(c, pos);
    }

    Unicode::size_type Unicode::rfind(self_view str, size_type pos) const {
        return view().rfind(str, pos);
    }

    Unicode::size_type Unicode::rfind(value_type c, size_type pos) const {
        return view().rfind(c, pos);
    }

    Unicode::size_type Unicode::find_first_of(self_view str, size_type pos) const {
        return view().find_first_of(str, pos);
    }

    Unicode::size_type Unicode::find_first_of(value_type c, size_type pos) const {
        return view().find_first_of(c, pos);
    }

    Unicode::size_type Unicode::find_last_of(self_view str, size_type pos) const {
        return view().find_last_of(str, pos);
    }

    Unicode::size_type Unicode::find_last_of(value_type c, size_type pos) const {
        return view().find_last_of(c, pos);
    }

    Unicode::size_type Unicode::find_first_not_of(self_view str, size_type pos) const {
        return view().find_first_not_of(str, pos);
    }

    Unicode::size_type Unicode::find_first_not_of(value_type c, size_type pos) const {
        return view().find_first_not_of(c, pos);
    }

    Unicode::size_type Unicode::find_last_not_of(self_view str, size_type pos) const {
        return view().find_last_not_of(str, pos);
    }

    Unicode::size_type Unicode::find_last_not_of(value_type c, size_type pos) const {
        return view().find_last_not_of(c, pos);
    }

    Unicode Unicode::substr(size_type pos, size_type n) const {
        return Unicode(view().substr(pos, n));
    }

    Unicode Unicode::concat(self_view lhs, self_view rhs) {
        Unicode ret;
        size_type lhs_size = lhs.size();
        size_type rhs_size = rhs.size();
        ret.reserve(lhs_size + rhs_size);
        ret.append(lhs);
        ret.append(rhs);
        return ret;
    }

    Unicode Unicode::concat(std::initializer_list<self_view> args) {
        size_t cap = 0;
        auto itr = args.begin();
        auto itr_end = args.end();
        for (; itr != itr_end; ++itr) {
            cap += itr->size();
        }
        Unicode ret;
        ret.resizeNoInit(cap);
        auto data = (Unicode::pointer)ret.data();
        for (itr = args.begin(); itr != itr_end; ++itr) {
            Unicode::traits_type::copy(data, itr->data(), itr->size());
            data += itr->size();
        }
        return ret;
    }

    Unicode Unicode::repeat(self_view sv, int64_t times) {
        times = TURBO_UNLIKELY(times < 0) ? 0 : times;
        auto result_size = times * sv.length();
        Unicode::ContainerType store(result_size, Unicode::ContainerType::NoInit{});
        auto* data = (Unicode::pointer)store.data();
        auto* src_data = sv.data();
        auto src_size = sv.size();
        for (int64_t i = 0; i < times; ++i) {
            Unicode::traits_type::copy(data, src_data, src_size);
            data += src_size;
        }
        return Unicode(std::move(store));
    }


    std::ostream &operator<<(std::ostream &out, const Unicode &input) {
        auto s = input.view().to_string();
        out.write(s.data(), s.size());
        return out;
    }

    /******************************************************************************
     * Unicode iterators
     *****************************************************************************/

    typename Unicode::iterator Unicode::Unicode::begin() {
        return data_.mutableData();
    }

    typename Unicode::const_iterator Unicode::Unicode::begin() const noexcept {
        return data_.data();
    }

    typename Unicode::const_iterator Unicode::Unicode::cbegin() const noexcept {
        return data_.data();
    }

    typename Unicode::iterator Unicode::Unicode::end() {
        return data_.mutableData() + data_.size();
    }

    typename Unicode::const_iterator Unicode::Unicode::end() const noexcept {
        return data_.data() + data_.size();
    }

    typename Unicode::const_iterator Unicode::Unicode::cend() const noexcept {
        return data_.data() + data_.size();
    }

    typename Unicode::reverse_iterator Unicode::Unicode::rbegin() {
        return reverse_iterator(end());
    }

    typename Unicode::const_reverse_iterator Unicode::Unicode::rbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    typename Unicode::const_reverse_iterator Unicode::Unicode::crbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    typename Unicode::reverse_iterator Unicode::Unicode::rend() {
        return reverse_iterator(begin());
    }

    typename Unicode::const_reverse_iterator Unicode::Unicode::rend() const noexcept {
        return const_reverse_iterator(begin());
    }

    typename Unicode::const_reverse_iterator Unicode::Unicode::crend() const noexcept {
        return const_reverse_iterator(begin());
    }

}  // namespace turbo
