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


#ifndef TURBO_CONCURRENT_THREAD_LOCAL_H_
#define TURBO_CONCURRENT_THREAD_LOCAL_H_

#include "turbo/meta/type_traits.h"
#include "turbo/concurrent/internal/thread_local_storage.h"
#include <cstdint>

namespace turbo {

    template<typename T>
    class ThreadLocal {
    public:
        ThreadLocal();

        template<typename... Args>
        explicit ThreadLocal(const Args &... args);

        ~ThreadLocal();

        ThreadLocal &operator =(const ThreadLocal &)= delete;

        ThreadLocal &operator =(ThreadLocal &&)= delete;

        ThreadLocal& operator =(const T & value);


    public:
        T &operator*() {
            return *get_impl();
        }
        
        T *operator->() {
            return get_impl();
        }
        
        const T &operator*() const {
            return *get_impl();
        }
        
        const T *operator->() const {
            return get_impl();
        }

        T &get() {
            return *get_impl();
        }

        const T &get() const {
            return *get_impl();
        }

        T *get_ptr() {
            return get_impl();
        }

        const T *get_ptr() const {
            return get_impl();
        }

        void reset() {
            T * ptr= get_impl();
            ptr->~T();
            new(ptr) T();
        }

        template<typename... Args>
        void reset(const Args &... args) {
            T * ptr= get_impl();
            ptr->~T();
            new(ptr) T(args...);
        }

        [[nodiscard]] uint32_t id() const {
            return _id;
        }

        
    private:
        T * get_impl();

        uint32_t _id{0};
    };

    template<typename T>
    ThreadLocal<T>::ThreadLocal() {
        _id = concurrent_internal::ThreadLocalStorage<T>::create_new_resource_id();
        T * ptr= get_impl();
        new(ptr) T();
    }

    template<typename T>
    ThreadLocal<T>::~ThreadLocal() {
        concurrent_internal::ThreadLocalStorage<T>::release_resource_id(_id);
    }
    template<typename T>
    template<typename... Args>
    ThreadLocal<T>::ThreadLocal(const Args &... args) {
        _id = concurrent_internal::ThreadLocalStorage<T>::create_new_resource_id();
        T * ptr= get_impl();
        new(ptr) T(args...);
    }


    template<typename T>
    T * ThreadLocal<T>::get_impl() {
        T * ptr= concurrent_internal::ThreadLocalStorage<T>::get_resource(_id);
        if(!ptr){
            ptr= concurrent_internal::ThreadLocalStorage<T>::get_or_create_resource(_id);
        }
        if(TURBO_UNLIKELY(!ptr)){
            TLOG_CRITICAL("ThreadLocal<T>::ThreadLocal(const Args &... args) failed to allocate memory for T");
            return nullptr;
        }
        return ptr;
    }

    template<typename T>
    ThreadLocal<T>& ThreadLocal<T>::operator =(const T & value) {
        auto ptr= get_impl();
        *ptr= value;
        return *this;
    }

} // namespace turbo
#endif  // TURBO_CONCURRENT_THREAD_LOCAL_H_
