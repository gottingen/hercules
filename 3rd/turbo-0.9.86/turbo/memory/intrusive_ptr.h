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
// Created by jeff on 24-1-10.
//

#ifndef TURBO_UTILITY_INTRUSIVE_PTR_H_
#define TURBO_UTILITY_INTRUSIVE_PTR_H_

#include <algorithm>

namespace turbo {
    namespace detail {

        //  Copyright (c) 2001, 2002 Peter Dimov
        //
        // Distributed under the Boost Software License, Version 1.0. (See
        // accompanying file LICENSE_1_0.txt or copy at
        // http://www.boost.org/LICENSE_1_0.txt)
        //
        //  See http://www.boost.org/libs/smart_ptr/intrusive_ptr.html for documentation.
        //
        //  intrusive_ptr
        //
        //  A smart pointer that uses intrusive reference counting.
        //
        //  Relies on unqualified calls to
        //
        //      void intrusive_ptr_add_ref(T * p);
        //      void intrusive_ptr_release(T * p);
        //
        //          (p != 0)
        //
        //  The object is responsible for destroying itself.

        template<class Y, class T>
        struct sp_convertible {
            typedef char (&yes)[1];
            typedef char (&no)[2];

            static yes f(T *);

            static no f(...);

            enum _vt {
                value = sizeof((f)(static_cast<Y *>(0))) == sizeof(yes)
            };
        };

        template<class Y, class T>
        struct sp_convertible<Y, T[]> {
            enum _vt {
                value = false
            };
        };
        template<class Y, class T>
        struct sp_convertible<Y[], T[]> {
            enum _vt {
                value = sp_convertible<Y[1], T[1]>::value
            };
        };
        template<class Y, std::size_t N, class T>
        struct sp_convertible<Y[N], T[]> {
            enum _vt {
                value = sp_convertible<Y[1], T[1]>::value
            };
        };

        struct sp_empty {
        };
        template<bool>
        struct sp_enable_if_convertible_impl;
        template<>
        struct sp_enable_if_convertible_impl<true> {
            typedef sp_empty type;
        };
        template<>
        struct sp_enable_if_convertible_impl<false> {
        };
        template<class Y, class T>
        struct sp_enable_if_convertible
                : public sp_enable_if_convertible_impl<sp_convertible<Y, T>::value> {
        };

    } // namespace detail

    template<class T>
    class intrusive_ptr {
    private:
        typedef intrusive_ptr this_type;
    public:
        typedef T element_type;

        intrusive_ptr() noexcept: px(0) {}

        intrusive_ptr(T *p, bool add_ref = true) : px(p) {
            if (px != 0 && add_ref) intrusive_ptr_add_ref(px);
        }

        template<class U>
        intrusive_ptr(const intrusive_ptr<U> &rhs,
                      typename detail::sp_enable_if_convertible<U, T>::type = detail::sp_empty())
                : px(rhs.get()) {
            if (px != 0) intrusive_ptr_add_ref(px);
        }

        intrusive_ptr(const intrusive_ptr &rhs) : px(rhs.px) {
            if (px != 0) intrusive_ptr_add_ref(px);
        }

        ~intrusive_ptr() {
            if (px != 0) intrusive_ptr_release(px);
        }

        template<class U>
        intrusive_ptr &operator=(const intrusive_ptr<U> &rhs) {
            this_type(rhs).swap(*this);
            return *this;
        }

        intrusive_ptr(intrusive_ptr &&rhs) noexcept: px(rhs.px) {
            rhs.px = 0;
        }

        intrusive_ptr &operator=(intrusive_ptr &&rhs) noexcept {
            this_type(static_cast< intrusive_ptr && >(rhs)).swap(*this);
            return *this;
        }

        intrusive_ptr &operator=(const intrusive_ptr &rhs) {
            this_type(rhs).swap(*this);
            return *this;
        }

        intrusive_ptr &operator=(T *rhs) {
            this_type(rhs).swap(*this);
            return *this;
        }

        void reset() noexcept {
            this_type().swap(*this);
        }

        void reset(T *rhs) {
            this_type(rhs).swap(*this);
        }

        void reset(T *rhs, bool add_ref) {
            this_type(rhs, add_ref).swap(*this);
        }

        T *get() const noexcept {
            return px;
        }

        T *detach() noexcept {
            T *ret = px;
            px = 0;
            return ret;
        }

        T &operator*() const {
            return *px;
        }

        T *operator->() const {
            return px;
        }

        explicit operator bool() const noexcept {
            return px != 0;
        }

        // operator! is redundant, but some compilers need it
        bool operator!() const noexcept {
            return px == 0;
        }

        void swap(intrusive_ptr &rhs) noexcept {
            T *tmp = px;
            px = rhs.px;
            rhs.px = tmp;
        }

    private:
        T *px;
    };

    template<class T, class U>
    inline bool operator==(const intrusive_ptr<T> &a, const intrusive_ptr<U> &b) {
        return a.get() == b.get();
    }

    template<class T, class U>
    inline bool operator!=(const intrusive_ptr<T> &a, const intrusive_ptr<U> &b) {
        return a.get() != b.get();
    }

    template<class T, class U>
    inline bool operator==(const intrusive_ptr<T> &a, U *b) {
        return a.get() == b;
    }

    template<class T, class U>
    inline bool operator!=(const intrusive_ptr<T> &a, U *b) {
        return a.get() != b;
    }

    template<class T, class U>
    inline bool operator==(T *a, const intrusive_ptr<U> &b) {
        return a == b.get();
    }

    template<class T, class U>
    inline bool operator!=(T *a, const intrusive_ptr<U> &b) {
        return a != b.get();
    }

    template<class T>
    inline bool operator==(const intrusive_ptr<T> &p, std::nullptr_t) noexcept {
        return p.get() == 0;
    }

    template<class T>
    inline bool operator==(std::nullptr_t, const intrusive_ptr<T> &p) noexcept {
        return p.get() == 0;
    }

    template<class T>
    inline bool operator!=(const intrusive_ptr<T> &p, std::nullptr_t) noexcept {
        return p.get() != 0;
    }

    template<class T>
    inline bool operator!=(std::nullptr_t, const intrusive_ptr<T> &p) noexcept {
        return p.get() != 0;
    }

    template<class T>
    inline bool operator<(const intrusive_ptr<T> &a, const intrusive_ptr<T> &b) {
        return std::less<T *>()(a.get(), b.get());
    }

    template<class T>
    void swap(intrusive_ptr<T> &lhs, intrusive_ptr<T> &rhs) {
        lhs.swap(rhs);
    }

    // mem_fn support

    template<class T>
    T *get_pointer(const intrusive_ptr<T> &p) {
        return p.get();
    }

    template<class T, class U>
    intrusive_ptr<T> static_pointer_cast(const intrusive_ptr<U> &p) {
        return static_cast<T *>(p.get());
    }

    template<class T, class U>
    intrusive_ptr<T> const_pointer_cast(const intrusive_ptr<U> &p) {
        return const_cast<T *>(p.get());
    }

    template<class T, class U>
    intrusive_ptr<T> dynamic_pointer_cast(const intrusive_ptr<U> &p) {
        return dynamic_cast<T *>(p.get());
    }

    template<class Y>
    std::ostream &operator<<(std::ostream &os, const intrusive_ptr<Y> &p) {
        os << p.get();
        return os;
    }

}  // namespace turbo

namespace turbo {
    template<typename H>
    H hash_value(H h, const turbo::intrusive_ptr<H> &p) {
        return H::combine(std::move(h), p.get());
    }
}
#endif  // TURBO_UTILITY_INTRUSIVE_PTR_H_
