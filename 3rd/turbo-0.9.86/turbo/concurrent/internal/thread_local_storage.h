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


#ifndef TURBO_CONCURRENT_THREAD_LOCAL_STORAGE_H_
#define TURBO_CONCURRENT_THREAD_LOCAL_STORAGE_H_

#include "turbo/meta/type_traits.h"
#include "turbo/base/reusable_id.h"
#include "turbo/log/logging.h"
#include "turbo/system/atexit.h"
#include <mutex>
#include <algorithm>

namespace turbo::concurrent_internal {

    template <typename T>
    class ThreadLocalStorage {
    public:
        typedef uint32_t resource_id_type;
        typedef T resource_type;
        typedef ReusableId<resource_type, resource_id_type> resource_id_generator_type;

        static constexpr resource_id_type BlockSize = 4096;

        static constexpr resource_id_type BLOCK_SIZE = BlockSize;

        static constexpr resource_id_type BLOCK_ITEM = (BLOCK_SIZE + sizeof(T) - 1) / sizeof(T);

        static constexpr resource_id_type INVALID_RESOURCE_ID = 32u * BLOCK_ITEM;

        struct Block {
            T *at(size_t i) {
                return _data + i;
            }
            T _data[BLOCK_ITEM];
        };

        static resource_id_type create_new_resource_id() {
            std::unique_lock l(_s_mutex);
            return resource_id_generator_type::get_instance().create_id().value_or(INVALID_RESOURCE_ID);
        }

        static void release_resource_id(resource_id_type id) {
            std::unique_lock l(_s_mutex);
            resource_id_generator_type::get_instance().free_id(id);
        }

        static T* get_resource(resource_id_type id) {
            if (TURBO_LIKELY(id < INVALID_RESOURCE_ID)) {
                if (_tls_blocks) {
                    auto block_id = id / BLOCK_ITEM;
                    if (block_id < _tls_blocks->size()) {
                        auto block = (*_tls_blocks)[block_id];
                        return block->at(id % BLOCK_ITEM);
                    }
                }

            }
            return nullptr;
        }

        static T* get_or_create_resource(resource_id_type id) {
            if (TURBO_UNLIKELY(id >= INVALID_RESOURCE_ID)) {
               TLOG_CRITICAL("invalid resource id: {}", id);
            }
            if(!_tls_blocks) {
                _tls_blocks = new std::vector<Block *>();
                turbo::thread_atexit(destroy_on_thread_exit);
            }
            uint32_t block_id = id / BLOCK_ITEM;
            if (block_id >= _tls_blocks->size()) {
                _tls_blocks->resize(std::max(block_id + 1, 32u));
            }

            auto block = (*_tls_blocks)[block_id];
            if (block == nullptr) {
                block = new Block();
                if(TURBO_UNLIKELY(block == nullptr)) {
                    TLOG_CRITICAL("Fail to create block, {}", strerror(errno));
                    return nullptr;
                }
                (*_tls_blocks)[block_id] = block;
            }

            return block->at(id % BLOCK_ITEM);
        }
    private:
        static void destroy_on_thread_exit() {
            if (!_tls_blocks) {
                return;
            }
            for (size_t i = 0; i < _tls_blocks->size(); ++i) {
                delete (*_tls_blocks)[i];
            }
            delete _tls_blocks;
            _tls_blocks = nullptr;
        }

        // guard for id generator
        static std::mutex                       _s_mutex;
        static __thread std::vector<Block *>  *_tls_blocks;
    };


    template <typename T>

    std::mutex ThreadLocalStorage<T>::_s_mutex;

    template <typename T>
    __thread std::vector<typename ThreadLocalStorage<T>::Block*> * ThreadLocalStorage<T>::_tls_blocks = nullptr;

}  // namespace turbo::concurrent_internal

#endif  // TURBO_CONCURRENT_THREAD_LOCAL_STORAGE_H_
