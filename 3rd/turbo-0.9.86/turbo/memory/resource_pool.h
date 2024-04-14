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


#ifndef TURBO_MEMORY_RESOURCE_POOL_H_
#define TURBO_MEMORY_RESOURCE_POOL_H_

#include <cstddef>
#include <memory>
#include <type_traits>
#include "turbo/status/status.h"
#include "turbo/platform/port.h"

namespace turbo {

    /**
     * @ingroup turbo_memory_pool
     * @brief ResourcePoolTraitsBase is the base class of ResourcePoolTraits,
     *        ResourcePoolTraitsBase is used to define the default value of ResourcePoolTraits
     *        Example:
     *        @code
     *
     *        struct MyStruct{
     *            int a;
     *            int b;
     *        };
     *
     *        template<>
     *        struct ResourcePoolTraits<MyStruct> : public ResourcePoolTraitsBase<T>{
     *              static constexpr size_t kBlockMaxSize = 1024 * 64;
     *              static constexpr size_t kBlockMaxItems = 256;
     *              static constexpr size_t kFreeChunkMaxItem = 256;
     *              static bool validate(const T *ptr) {
     *                  return ptr->a == 1;
     *              }
     *        };
     *
     *        @endcode
     *
     * @tparam T the type of object
     */
    template<typename T>
    struct ResourcePoolTraitsBase {
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
     * @brief ResourcePoolTraits is used to define redefine the default value of ResourcePoolTraitsBase
     *        Example:
     *        @code
     *
     *        struct MyStruct{
     *            int a;
     *            int b;
     *        };
     *
     *        template<>
     *        struct ResourcePoolTraits<MyStruct> : public ResourcePoolTraitsBase<T>{
     *              static constexpr size_t kBlockMaxSize = 1024 * 64;
     *              static constexpr size_t kBlockMaxItems = 256;
     *              static constexpr size_t kFreeChunkMaxItem = 256;
     *              static bool validate(const T *ptr) {
     *                  return ptr->a == 1;
     *              }
     *        };
     *
     *        @endcode
     *
     * @tparam T the type of object
     */
    template<typename T>
    struct ResourcePoolTraits : public ResourcePoolTraitsBase<T> {

    };

    struct ResourcePoolInfo {
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

#include "turbo/memory/internal/resource_pool_impl.h"

namespace turbo {


    /**
     * @ingroup turbo_memory_pool
     * @brief Get an object typed |T| and write its identifier into |id|.
     *        The object should be cleared before usage.
     *        NOTE: T must be default-constructible.
     * @tparam T the type of object
     * @param id [output]the identifier of object
     * @return the object
     */
    template<typename T>
    inline T *get_resource(ResourceId<T> *id) {
        return ResourcePool<T>::singleton()->get_resource(id);
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief Get an object typed |T| and write its identifier into |id|.
     *        The object should be cleared before usage.
     *        Example:
     *        @code
     *        struct MyStruct{
     *           int a;
     *           int b;
     *           MyStruct(int a, int b) : a(a), b(b) {}
     *       };
     *       ResourceId<MyStruct> id;
     *       MyStruct *ptr = get_resource(&id, 1, 2);
     *       @endcode
     * @note T must have a constructor with the same number of parameters as Args.
     * @tparam T the type of object
     * @tparam Args the type of parameters
     * @param id [output]the identifier of object
     * @return the object
     */
    template<typename T, typename ...Args>
    inline T *get_resource(ResourceId<T> *id, const Args &...args) {
        return ResourcePool<T>::singleton()->get_resource(id, args...);
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief Get an object typed |T| and write its identifier into |id|.
     *        The object should be cleared before usage.
     *        Example:
     *        @code
     *        struct MyStruct{
     *           std::string a;
     *           std::string b;
     *        };
     *        ResourceId<MyStruct> id;
     *        std::string a = "hello";
     *        std::string b = "world";
     *        MyStruct *ptr = get_resource(&id, std::move(a), std::move(b));
     *        @endcode
     *        The above code will get an object typed MyStruct with arguments a and b.
     * @note  the constructor of T must have a parameter of type Args&&. after the call, a and b will be empty.
     * @tparam T the type of object
     * @tparam Args the type of parameters
     * @param id [output]the identifier of object
     * @return the object
     */
    template<typename T, typename ...Args>
    inline T *get_resource(ResourceId<T> *id, Args &&...arg) {
        return ResourcePool<T>::singleton()->get_resource(id, std::forward<Args>(arg)...);
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief Return the object associated with identifier |id| back. The object is NOT
     *        destructed and will be returned by later get_resource<T>. Similar with
     *        free/delete, validity of the id is not checked, user shall not return a
     *        not-yet-allocated or already-returned id otherwise behavior is undefined.
     * @tparam T the type of object
     * @param id the identifier of object
     * @return int 0 when successful, -1 otherwise.
     */
    template<typename T>
    inline int return_resource(ResourceId<T> id) {
        return ResourcePool<T>::singleton()->return_resource(id);
    }

    // Get the object associated with the identifier |id|.
    // Returns NULL if |id| was not allocated by get_resource<T> or
    // ResourcePool<T>::get_resource() of a variant before.
    // Addressing a free(returned to pool) identifier does not return NULL.
    // NOTE: Calling this function before any other get_resource<T>/
    //       return_resource<T>/address<T>, even if the identifier is valid,
    //       may race with another thread calling clear_resources<T>.

    /**
     * @ingroup turbo_memory_pool
     * @brief Get the object associated with the identifier |id|.
     *        Returns NULL if |id| was not allocated by get_resource<T> or
     *        ResourcePool<T>::get_resource() of a variant before.
     *        Addressing a free(returned to pool) identifier does not return NULL.
     *        NOTE: Calling this function before any other get_resource<T>/
     *        return_resource<T>/address<T>, even if the identifier is valid,
     *        may race with another thread calling clear_resources<T>.
     * @tparam T the type of object
     * @param id the identifier of object
     * @return the object
     */
    template<typename T>
    inline T *address_resource(ResourceId<T> id) {
        return ResourcePool<T>::address_resource(id);
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief Reclaim all allocated resources typed T if caller is the last thread called
     *        this function, otherwise do nothing. You rarely need to call this function
     *        manually because it's called automatically when each thread quits.
     * @tparam T the type of object
     */
    template<typename T>
    inline void clear_resources() {
        ResourcePool<T>::singleton()->clear_resources();
    }

    /**
     * @ingroup turbo_memory_pool
     * @brief Get description of resources typed T.
     *        This function is possibly slow because it iterates internal structures.
     *        Don't use it frequently like a "getter" function.
     * @tparam T the type of object
     * @return ResourcePoolInfo
     */
    template<typename T>
    ResourcePoolInfo describe_resources() {
        return ResourcePool<T>::singleton()->describe_resources();
    }

    template<typename T>
    uint64_t make_resource_id(uint32_t index, ResourceId<T> rid) {
        return (static_cast<uint64_t>(index) << 32) | rid.value;
    }

    template<typename T>
    ResourceId<T> get_resource_id(uint64_t id) {
        return ResourceId<T>{static_cast<uint32_t>(id & 0xFFFFFFFF)};
    }

}  // namespace turbo

#endif  // TURBO_MEMORY_RESOURCE_POOL_H_
