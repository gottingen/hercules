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

#ifndef TURBO_CONCURRENT_LINKED_THREAD_LOCAL_H_
#define TURBO_CONCURRENT_LINKED_THREAD_LOCAL_H_

#include "turbo/container/intrusive_list.h"
#include "turbo/meta/type_traits.h"
#include "turbo/concurrent/internal/thread_local_storage.h"

namespace turbo {
    template<typename T>
    class WriteLocal;

    template<typename T, typename Enabler = void>
    class Element {
    public:

        using is_sequentail_type = std::false_type;

        void load(T *out) {
            std::unique_lock guard(_lock);
            *out = _value;
        }

        void store(const T &new_value) {
            std::unique_lock guard(_lock);
            _value = new_value;
        }

        void exchange(T *prev, const T &new_value) {
            std::unique_lock guard(_lock);
            *prev = _value;
            _value = new_value;
        }

        template<typename Op, typename T1>
        void modify(const Op &op, const T1 &value2) {
            std::unique_lock guard(_lock);
            call_op_returning_void(op, _value, value2);
        }

        // [Unique]
        template<typename Op, typename GlobalValue>
        void merge_global(const Op &op, GlobalValue &global_value) {
            _lock.lock();
            op(global_value, _value);
            _lock.unlock();
        }

    private:
        T _value;
        std::mutex _lock;
    };

    template<typename T>
    class Element<T, typename std::enable_if<is_atomical<T>::value>::type> {
    public:
        using is_sequentail_type = std::false_type;

        inline void load(T *out) {
            *out = _value.load(std::memory_order_relaxed);
        }

        inline void store(T new_value) {
            _value.store(new_value, std::memory_order_relaxed);
        }

        inline void exchange(T *prev, T new_value) {
            *prev = _value.exchange(new_value, std::memory_order_relaxed);
        }

        // [Unique]
        inline bool compare_exchange_weak(T &expected, T new_value) {
            return _value.compare_exchange_weak(expected, new_value,
                                                std::memory_order_relaxed);
        }

        template<typename Op, typename T1>
        void modify(const Op &op, const T1 &value2) {
            T old_value = _value.load(std::memory_order_relaxed);
            T new_value = old_value;
            call_op_returning_void(op, new_value, value2);
            // There's a contention with the reset operation of combiner,
            // if the tls value has been modified during _op, the
            // compare_exchange_weak operation will fail and recalculation is
            // to be processed according to the new version of value
            while (!_value.compare_exchange_weak(
                    old_value, new_value, std::memory_order_relaxed)) {
                new_value = old_value;
                call_op_returning_void(op, new_value, value2);
            }
        }

    private:
        std::atomic<T> _value;
    };


    /**
     * @brief ThreadLocal
     * @undecoded
     *                         T1            T2        ....    Tn
     *                         \             |         |       /
     *                          \            |         |      /
     *                           \          |         |     /
     *                            \         |         |    /
     *                             \        |         |   /        <-----------------multi-thread write and read
     *                              \       |         |  /
     *                               --------------------
     *                               | memory barrier |
     *                               ------------------
     *-----------------------------------------------------------------------------------------------------------------
     *
     *                               T1     T2        ....    Tn
     *                               |       |         |       |
     *                               |       |         |       |      <------------------write variables
     *                               |       |         |       |
     *                            ______  _______  _______  _______
     *                            | mem || mem  || mem  |  | mem  |
     *                            |  1  ||  2   ||  3   |  |  n   |
     *                            |____| |______||_______| |______|
     *                             \       |         |       /
     *                              \      |         |      /                    <---------- read variables
     *                                \    |         |     /
     *                                ______________________
     *                                | read memory barrier |   reader is a single thread
     *                                ----------------------
     *
     *  @endundecoded
     * @tparam T
     */
    template<typename T>
    class WriteLocal {
    public:
        struct ElementHolder : public intrusive_list_node {
            ElementHolder() : intrusive_list_node() {}

            ~ElementHolder() {
                if (holder) {
                    holder->remove_element(this);
                }
            }

            Element<T> element;
            WriteLocal *holder{nullptr};
        };

    public:
        WriteLocal() {
            _id = concurrent_internal::ThreadLocalStorage<T>::create_new_resource_id();
            ElementHolder *ptr = get_impl();
            ptr->element.store(T());
            ptr->holder = this;
            {
                std::unique_lock lock(g_mutex);
                g_list.push_back(*ptr);
            }
        }

        ~WriteLocal() {
            ElementHolder *ptr = get_impl();
            {
                std::unique_lock lock(g_mutex);
                intrusive_list<ElementHolder>::remove(*ptr);
                ptr->holder = nullptr;
            }
            concurrent_internal::ThreadLocalStorage<T>::release_resource_id(_id);
        }

        void remove_element(ElementHolder *element) {
            std::unique_lock lock(g_mutex);
            intrusive_list<ElementHolder>::remove(*element);
        }

        WriteLocal(const WriteLocal &) = delete;

        WriteLocal &operator=(const WriteLocal &) = delete;

        void set(const T &new_value) {
            ElementHolder *ptr = get_impl();
            ptr->element.store(new_value);
        }

        T get() {
            ElementHolder *ptr = get_impl();
            T value;
            ptr->element.load(&value);
            return value;
        }

        void exchange(T *prev, const T &new_value) {
            ElementHolder *ptr = get_impl();
            ptr->element.exchange(prev, new_value);
        }

        void merge_global(const T &new_value) {
            std::unique_lock lock(g_mutex);
            for (auto &holder: g_list) {
                holder.element.store(new_value);
            }
        }

        void merge_global(const T &new_value, const std::function<bool(const T &)> &filter) {
            std::unique_lock lock(g_mutex);
            for (auto &holder: g_list) {
                if (filter(holder.element)) {
                    holder.element.store(new_value);
                }
            }
        }

        void list(const std::function<void(const T &)> &callback) {
            std::unique_lock lock(g_mutex);
            for (auto &holder: g_list) {
                T value;
                holder.element.load(&value);
                callback(value);
            }
        }

        void list(const std::function<void(const T &)> &callback, const std::function<bool(const T &)> &filter) {
            std::unique_lock lock(g_mutex);
            for (auto &holder: g_list) {
                T value;
                holder.element.load(&value);
                if (filter(value)) {
                    callback(value);
                }
            }
        }

        void list(std::vector<T> &values) {
            std::unique_lock lock(g_mutex);
            for (auto &holder: g_list) {
                T value;
                holder.element.load(&value);
                values.push_back(value);
            }
        }

    private:
        ElementHolder *get_impl() {
            ElementHolder *ptr = concurrent_internal::ThreadLocalStorage<ElementHolder>::get_resource(_id);
            if (!ptr) {
                ptr = concurrent_internal::ThreadLocalStorage<ElementHolder>::get_or_create_resource(_id);
                if (TURBO_UNLIKELY(!ptr)) {
                    TLOG_CRITICAL("ThreadLocal<T>::ThreadLocal(const Args &... args) failed to allocate memory for T");
                    return nullptr;
                }
                {
                    std::unique_lock lock(g_mutex);
                    ptr->holder = this;
                    g_list.push_back(*ptr);
                }
            }
            return ptr;
        }

    private:
        uint32_t _id{0};
        intrusive_list<ElementHolder> g_list;
        std::mutex g_mutex;
    };

}  // namespace turbo
#endif  // TURBO_CONCURRENT_LINKED_THREAD_LOCAL_H_
