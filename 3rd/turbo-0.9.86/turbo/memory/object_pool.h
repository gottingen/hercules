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


#ifndef TURBO_MEMORY_OBJECT_POOL_H_
#define TURBO_MEMORY_OBJECT_POOL_H_

#include <cstddef>
#include <memory>
#include "turbo/status/status.h"
#include "turbo/platform/port.h"

/**
 * @brief
 * @file object_pool.h
 * @date 2021-08-31
 * @version 1.0
 * @note
 * @history
 * 2021-08-31: 1.0 created by jeff
 */

namespace turbo {

    /**
     * @ingroup turbo_memory_pool
     * @brief ObjectPoolTraitsBase is the base class of ObjectPoolTraits
     *        ObjectPoolTraitsBase is used to define the default value of ObjectPoolTraits
     * @tparam T the type of object
     */
    template<typename T>
    struct ObjectPoolTraitsBase {
        static constexpr size_t kBlockMaxSize = 1024 * 64;

        static constexpr size_t kBlockMaxItems = 256;

        static constexpr size_t kFreeChunkMaxItem = 256;

        static constexpr size_t block_max_size() {
            return kBlockMaxSize;
        }

        static constexpr size_t block_max_items() {
            return kBlockMaxItems;
        }

        static constexpr size_t free_chunk_max_items() {
            return block_max_size() / sizeof(T) >= kFreeChunkMaxItem ? kFreeChunkMaxItem
                                                                     : block_max_size() / sizeof(T);
        }

        static constexpr bool validate(const T *ptr) {
            return true;
        }
    private:
        static_assert(kBlockMaxSize >= sizeof(T), "kBlockMaxSize must be greater than sizeof(T)");
        static_assert(kBlockMaxItems > 0, "kBlockMaxItems must be greater than 0");
        static_assert(kFreeChunkMaxItem >= 0, "kFreeChunkMaxSize must be greater than 0");
    };

    /**
     * @ingroup turbo_memory_pool
     * @brief ObjectPoolTraits is used to define redefine the default value of ObjectPoolTraitsBase
     *        Example:
     *        @code
     *        struct MyObject {
     *          int a;
     *          int b;
     *        };
     *
     *        struct ObjectPoolTraits<MyObject> : public ObjectPoolTraitsBase<MyObject> {
     *            typedef ObjectPoolTraitsBase<MyObject> Base;
     *            using Base::kBlockMaxSize;
     *            using Base::kBlockMaxItems;
     *            KFreeChunkMaxItem = 2048;
     *            using Base::validate;
     *        };
     *        @endcode
     *        The above code will redefine the default value of ObjectPoolTraitsBase
     *        kFreeChunkMaxItem = 2048
     * @tparam T the type of object
     */
    template<typename T>
    struct ObjectPoolTraits : public ObjectPoolTraitsBase<T>{
        
    };

    struct ObjectPoolInfo {
        size_t local_pool_num{0};
        size_t block_group_num{0};
        size_t block_num{0};
        size_t item_num{0};
        size_t block_item_num{0};
        size_t free_chunk_item_num{0};
        size_t total_size{0};
        size_t free_item_num{0};
    };
}  // namespace turbo
#include "turbo/memory/internal/object_pool_impl.h"

namespace turbo {

    // Get an object typed |T|. The object should be cleared before usage.
    // NOTE: T must be default-constructible.

    /**
     * @ingroup turbo_memory_pool
     * @brief get_object is used to get an object typed |T|. The object should be cleared before usage.
     *        NOTE: T must have a default constructor.
     *        Example:
     *        @code
     *        struct MyObject {
     *            int a;
     *            int b;
     *            MyObject() : a(0), b(0) {}
     *            void clear() {
     *                a = 0;
     *                b = 0;
     *                // other clear operation
     *                // ...
     *             }
     *         };
     *
     *         MyObject *obj = get_object<MyObject>();
     *         obj->clear();
     *         @endcode
     *         The above code will get an object typed MyObject, and the object should be cleared before usage.
     * @note MyObject must have a default constructor.
     * @return T* the object typed T
     */
    template <typename T>
    inline T* get_object() {
        return ObjectPool<T>::singleton()->get_object();
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief get_object is used to get an object typed |T| with arguments.
     *        Example:
     *        @code
     *        struct MyObject {
     *            int a;
     *            int b;
     *            MyObject(int a, int b) : a(a), b(b) {}
     *            void clear() {
     *                a = 0;
     *                b = 0;
     *                // other clear operation
     *                // ...
     *             }
     *         };
     *
     *         MyObject *obj = get_object<MyObject>(1, 2);
     *         obj->clear();
     *         @endcode
     *         The above code will get an object typed MyObject with arguments 1 and 2.
     * @return T* the object typed T
     */
    template <typename T, typename ...Args>
    inline T* get_object(const Args &...args) {
        return ObjectPool<T>::singleton()->get_object(args...);
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief get_object is used to get an object typed |T| with arguments.
     *        Example:
     *        @code
     *        struct MyObject {
     *            std::string a;
     *            int b;
     *            MyObject(std::string&& a, int b) : a(std::move(a)), b(b) {}
     *
     *            void clear() {
     *                a.clear();
     *                b = 0;
     *                // other clear operation
     *            }
     *        };
     *
     *        std::string a = "hello";
     *        MyObject *obj = get_object<MyObject>(std::move(a), 2);
     *        // a is empty now
     *        @endcode
     *        The above code will get an object typed MyObject with arguments lvalue a and 2.
     * @return T* the object typed T
     * @note the arguments will be moved to the object
     */
    template <typename T, typename ...Args>
    inline T* get_object(Args &&...args) {
        return ObjectPool<T>::singleton()->get_object(std::forward<Args>(args)...);
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief return_object is used to return the object |ptr| back. The object is NOT destructed and will be
     *        returned by later get_object<T>. Similar with free/delete, validity of
     *        the object is not checked, user shall not return a not-yet-allocated or
     *        already-returned object otherwise behavior is undefined.
     *        @param ptr the object to be returned
     *        @return int 0 when successful, -1 otherwise.
     */
    template <typename T> inline int return_object(T* ptr) {
        return ObjectPool<T>::singleton()->return_object(ptr);
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief reclaim_objects is used to reclaim all allocated objects typed T if caller is the last thread called
     *        this function, otherwise do nothing. You rarely need to call this function
     *        manually because it's called automatically when each thread quits.
     */
    template <typename T> inline void clear_objects() {
        ObjectPool<T>::singleton()->clear_objects();
    }

    // Get description of objects typed T.
    // This function is possibly slow because it iterates internal structures.
    // Don't use it frequently like a "getter" function.

    /**
     * @ingroup turbo_memory_pool
     * @brief Get description of objects typed T.
     *        This function is possibly slow because it iterates internal structures.
     *        Don't use it frequently like a "getter" function.
     */
    template <typename T>
    ObjectPoolInfo describe_objects() {
        return ObjectPool<T>::singleton()->describe_objects();
    }

}  // namespace turbo

#endif  // TURBO_MEMORY_OBJECT_POOL_H_
