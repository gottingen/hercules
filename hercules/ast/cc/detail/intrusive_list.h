// Copyright 2024 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#pragma once

#include <iterator>
#include <memory>
#include <collie/type_safe/optional_ref.h>
#include <hercules/ast/cc/cppast_fwd.h>
#include <hercules/ast/cc/detail/assert.h>

namespace hercules::ccast::detail {

    template<typename T>
    class intrusive_list_node {
    public:
        intrusive_list_node() = default;

        intrusive_list_node(intrusive_list_node &&) = default;

        intrusive_list_node &operator=(intrusive_list_node &&) = default;

        ~intrusive_list_node() noexcept {
            // Free iteratively to avoid stack overflow in debug builds.
            auto next = next_.release();
            while (next) {
                std::unique_ptr<T> cur(next);
                next = cur->next_.release();
            }
        }

    private:
        void do_on_insert(const T &parent) noexcept {
            static_cast<T &>(*this).on_insert(parent);
        }

        std::unique_ptr<T> next_;

        template<typename U>
        friend
        struct intrusive_list_access;
    };

    template<typename T>
    struct intrusive_list_access {
        template<typename U>
        static T *get_next(const U &obj) {
            static_assert(std::is_base_of<U, T>::value, "must be a base");
            return static_cast<T *>(obj.next_.get());
        }

        template<typename U>
        static T *set_next(U &obj, std::unique_ptr<T> node) {
            static_assert(std::is_base_of<U, T>::value, "must be a base");
            obj.next_ = std::move(node);
            return static_cast<T *>(obj.next_.get());
        }

        template<typename U, typename V>
        static void on_insert(U &obj, const V &parent) {
            obj.do_on_insert(parent);
        }
    };

    template<typename T>
    class intrusive_list_iterator {
    public:
        using value_type = T;
        using reference = T &;
        using pointer = T *;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        intrusive_list_iterator() noexcept: cur_(nullptr) {}

        reference operator*() const noexcept {
            return *cur_;
        }

        pointer operator->() const noexcept {
            return cur_;
        }

        intrusive_list_iterator &operator++() noexcept {
            cur_ = intrusive_list_access<T>::get_next(*cur_);
            return *this;
        }

        intrusive_list_iterator operator++(int) noexcept {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const intrusive_list_iterator &a,
                               const intrusive_list_iterator &b) noexcept {
            return a.cur_ == b.cur_;
        }

        friend bool operator!=(const intrusive_list_iterator &a,
                               const intrusive_list_iterator &b) noexcept {
            return !(a == b);
        }

    private:
        intrusive_list_iterator(T *ptr) : cur_(ptr) {}

        T *cur_;

        template<typename U>
        friend
        class intrusive_list;
    };

    template<typename T>
    class intrusive_list {
    public:
        intrusive_list() = default;

        //=== modifiers ===//
        template<typename Dummy = T,
                typename = typename std::enable_if<std::is_same<Dummy, cpp_file>::value>::type>
        void push_back(std::unique_ptr<T> obj) noexcept {
            push_back_impl(std::move(obj));
        }

        template<typename U,
                typename = typename std::enable_if<!std::is_same<T, cpp_file>::value, U>::type>
        void push_back(const U &parent, std::unique_ptr<T> obj) noexcept {
            push_back_impl(std::move(obj));
            intrusive_list_access<T>::on_insert(last_.value(), parent);
        }

        //=== accesors ===//
        bool empty() const noexcept {
            return first_ == nullptr;
        }

        collie::ts::optional_ref <T> front() noexcept {
            return collie::ts::opt_ref(first_.get());
        }

        collie::ts::optional_ref<const T> front() const noexcept {
            return collie::ts::opt_cref(first_.get());
        }

        collie::ts::optional_ref <T> back() noexcept {
            return last_;
        }

        collie::ts::optional_ref<const T> back() const noexcept {
            return last_;
        }

        //=== iterators ===//
        using iterator = intrusive_list_iterator<T>;
        using const_iterator = intrusive_list_iterator<const T>;

        iterator begin() noexcept {
            return iterator(first_.get());
        }

        iterator end() noexcept {
            return {};
        }

        const_iterator begin() const noexcept {
            return const_iterator(first_.get());
        }

        const_iterator end() const noexcept {
            return {};
        }

    private:
        void push_back_impl(std::unique_ptr<T> obj) {
            DEBUG_ASSERT(obj != nullptr, detail::assert_handler{});

            if (last_) {
                auto ptr = intrusive_list_access<T>::set_next(last_.value(), std::move(obj));
                last_ = collie::ts::ref(*ptr);
            } else {
                first_ = std::move(obj);
                last_ = collie::ts::opt_ref(first_.get());
            }
        }

        std::unique_ptr<T> first_;
        collie::ts::optional_ref <T> last_;
    };

    template<typename T>
    class iteratable_intrusive_list {
    public:
        iteratable_intrusive_list(collie::ts::object_ref<const intrusive_list<T>> list) : list_(list) {}

        bool empty() const noexcept {
            return list_->empty();
        }

        using iterator = typename intrusive_list<T>::const_iterator;

        iterator begin() const noexcept {
            return list_->begin();
        }

        iterator end() const noexcept {
            return list_->end();
        }

    private:
        collie::ts::object_ref<const intrusive_list<T>> list_;
    };
} // namespace hercules::ccast::detail
