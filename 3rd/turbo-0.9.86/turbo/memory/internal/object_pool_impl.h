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

#ifndef TURBO_MEMORY_OBJECT_POOL_IMPL_H_
#define TURBO_MEMORY_OBJECT_POOL_IMPL_H_


#include <iostream>
#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include "turbo/system/atexit.h"

namespace turbo {

    template<typename T, size_t NITEM>
    struct ObjectPoolFreeChunk {
        size_t nfree;
        T *ptrs[NITEM];
    };
    // for gcc 3.4.5
    template<typename T>
    struct ObjectPoolFreeChunk<T, 0> {
        size_t nfree;
        T *ptrs[0];
    };


    static const size_t OP_MAX_BLOCK_NGROUP = 65536;
    static const size_t OP_GROUP_NBLOCK_NBIT = 16;
    static const size_t OP_GROUP_NBLOCK = (1UL << OP_GROUP_NBLOCK_NBIT);
    static const size_t OP_INITIAL_FREE_LIST_SIZE = 1024;


    template<typename T>
    class TURBO_CACHE_LINE_ALIGNED ObjectPool {
    public:
        static constexpr size_t BLOCK_NITEM = ObjectPoolTraits<T>::block_max_items();
        static constexpr size_t FREE_CHUNK_NITEM = BLOCK_NITEM;

        // Free objects are batched in a FreeChunk before they're added to
        // global list(_free_chunks).
        typedef ObjectPoolFreeChunk<T, FREE_CHUNK_NITEM> FreeChunk;
        typedef ObjectPoolFreeChunk<T, 0> DynamicFreeChunk;

        // When a thread needs memory, it allocates a Block. To improve locality,
        // items in the Block are only used by the thread.
        // To support cache-aligned objects, align Block.items by cacheline.
        struct TURBO_CACHE_LINE_ALIGNED Block {
            char items[sizeof(T) * BLOCK_NITEM]{};
            size_t nitem{0};
        };

        // An Object addresses at most OP_MAX_BLOCK_NGROUP BlockGroups,
        // each BlockGroup addresses at most OP_GROUP_NBLOCK blocks. So an
        // object addresses at most OP_MAX_BLOCK_NGROUP * OP_GROUP_NBLOCK Blocks.
        struct BlockGroup {
            std::atomic<size_t> nblock;
            std::atomic<Block *> blocks[OP_GROUP_NBLOCK];

            BlockGroup() : nblock(0) {
                // We fetch_add nblock in add_block() before setting the entry,
                // thus address_resource() may sees the unset entry. Initialize
                // all entries to nullptr makes such address_resource() return nullptr.
                memset(static_cast<void *>(blocks), 0, sizeof(std::atomic<Block *>) * OP_GROUP_NBLOCK);
            }
        };

        // Each thread has an instance of this class.
        class TURBO_CACHE_LINE_ALIGNED LocalPool {
        public:
            explicit LocalPool(ObjectPool *pool)
                    : _pool(pool), _cur_block(nullptr), _cur_block_index(0) {
                _cur_free.nfree = 0;
            }

            ~LocalPool() {
                // Add to global _free if there're some free objects
                if (_cur_free.nfree) {
                    _pool->push_free_chunk(_cur_free);
                }

                _pool->clear_from_destructor_of_local_pool();
            }

            static void delete_local_pool(void *arg) {
                delete (LocalPool *) arg;
            }

            inline T *get_raw() {
                /* Fetch local free ptr */
                if (_cur_free.nfree) {
                    _global_nfree.fetch_sub(1, std::memory_order_relaxed);
                    return _cur_free.ptrs[--_cur_free.nfree];
                }
                /* Fetch a FreeChunk from global.
                   TODO: Popping from _free needs to copy a FreeChunk which is
                   costly, but hardly impacts amortized performance. */
                if (_pool->pop_free_chunk(_cur_free)) {
                    _global_nfree.fetch_sub(1, std::memory_order_relaxed);
                    return _cur_free.ptrs[--_cur_free.nfree];
                }
                /* Fetch memory from local block */
                if (_cur_block && _cur_block->nitem < BLOCK_NITEM) {
                    T *obj = (T *) _cur_block->items + _cur_block->nitem;
                    ++_cur_block->nitem;
                    return obj;
                }
                /* Fetch a Block from global */
                _cur_block = add_block(&_cur_block_index);
                if (_cur_block != nullptr) {
                    T *obj = (T *) _cur_block->items + _cur_block->nitem;
                    ++_cur_block->nitem;
                    return obj;
                }
                return nullptr;
            }

            inline T *get_free_list() {
                /* Fetch local free ptr */
                if (_cur_free.nfree) {
                    _global_nfree.fetch_sub(1, std::memory_order_relaxed);
                    return _cur_free.ptrs[--_cur_free.nfree];
                }
                /* Fetch a FreeChunk from global.
                   TODO: Popping from _free needs to copy a FreeChunk which is
                   costly, but hardly impacts amortized performance. */
                if (_pool->pop_free_chunk(_cur_free)) {
                    _global_nfree.fetch_sub(1, std::memory_order_relaxed);
                    return _cur_free.ptrs[--_cur_free.nfree];
                }
                return nullptr;
            }

            inline T *get() {
                auto ptr = get_free_list();
                if (ptr != nullptr) {
                    return ptr;
                }
                if (_cur_block && _cur_block->nitem < BLOCK_NITEM) {
                    T *obj = (T *) _cur_block->items + _cur_block->nitem;
                    new(obj) T();
                    if (!ObjectPoolTraits<T>::validate(obj)) {
                        obj->~T();
                        return nullptr;
                    }
                    ++_cur_block->nitem;
                    return obj;
                }
                /* Fetch a Block from global */
                _cur_block = add_block(&_cur_block_index);
                if (_cur_block != nullptr) {
                    T *obj = (T *) _cur_block->items + _cur_block->nitem;
                    new(obj) T();
                    if (!ObjectPoolTraits<T>::validate(obj)) {
                        obj->~T();
                        return nullptr;
                    }
                    ++_cur_block->nitem;
                    return obj;
                }
                return nullptr;
            }

            template<typename ...Args>
            inline T *get(const Args &...args) {
                auto ptr = get_free_list();
                if (ptr != nullptr) {
                    return ptr;
                }
                if (_cur_block && _cur_block->nitem < BLOCK_NITEM) {
                    T *obj = (T *) _cur_block->items + _cur_block->nitem;
                    new(obj) T(args...);
                    if (!ObjectPoolTraits<T>::validate(obj)) {
                        obj->~T();
                        return nullptr;
                    }
                    ++_cur_block->nitem;
                    return obj;
                }
                _cur_block = add_block(&_cur_block_index);
                if (_cur_block != nullptr) {
                    T *obj = (T *) _cur_block->items + _cur_block->nitem;
                    new(obj) T(args...);
                    if (!ObjectPoolTraits<T>::validate(obj)) {
                        obj->~T();
                        return nullptr;
                    }
                    ++_cur_block->nitem;
                    return obj;
                }
                return nullptr;
            }

            template<typename ...Args>
            inline T *get(Args &&...args) {
                auto ptr = get_free_list();
                if (ptr != nullptr) {
                    return ptr;
                }
                if (_cur_block && _cur_block->nitem < BLOCK_NITEM) {
                    T *obj = (T *) _cur_block->items + _cur_block->nitem;
                    new(obj) T(std::forward<Args>(args)...);
                    if (!ObjectPoolTraits<T>::validate(obj)) {
                        obj->~T();
                        return nullptr;
                    }
                    ++_cur_block->nitem;
                    return obj;
                }
                _cur_block = add_block(&_cur_block_index);
                if (_cur_block != nullptr) {
                    T *obj = (T *) _cur_block->items + _cur_block->nitem;
                    new(obj) T(std::forward<Args>(args)...);
                    if (!ObjectPoolTraits<T>::validate(obj)) {
                        obj->~T();
                        return nullptr;
                    }
                    ++_cur_block->nitem;
                    return obj;
                }
                return nullptr;
            }

            inline int return_object(T *ptr) {
                // Return to local free list
                if (_cur_free.nfree < ObjectPool::free_chunk_nitem()) {
                    _cur_free.ptrs[_cur_free.nfree++] = ptr;
                    _global_nfree.fetch_add(1, std::memory_order_relaxed);
                    return 0;
                }
                // Local free list is full, return it to global.
                // For copying issue, check comment in upper get()
                if (_pool->push_free_chunk(_cur_free)) {
                    _cur_free.nfree = 1;
                    _cur_free.ptrs[0] = ptr;
                    _global_nfree.fetch_add(1, std::memory_order_relaxed);
                    return 0;
                }
                return -1;
            }

        private:
            ObjectPool *_pool;
            Block *_cur_block;
            size_t _cur_block_index;
            FreeChunk _cur_free;
        };

        inline T *get_object() {
            LocalPool *lp = get_or_new_local_pool();
            if (TURBO_LIKELY(lp != nullptr)) {
                return lp->get();
            }
            return nullptr;
        }

        template<typename ...Args>
        inline T *get_object(const Args &...args) {
            LocalPool *lp = get_or_new_local_pool();
            if (TURBO_LIKELY(lp != nullptr)) {
                return lp->get(args...);
            }
            return nullptr;
        }

        template<typename ...Args>
        inline T *get_object(Args &&...args) {
            LocalPool *lp = get_or_new_local_pool();
            if (TURBO_LIKELY(lp != nullptr)) {
                return lp->get(std::forward<Args>(args)...);
            }
            return nullptr;
        }

        inline int return_object(T *ptr) {
            LocalPool *lp = get_or_new_local_pool();
            if (TURBO_LIKELY(lp != nullptr)) {
                return lp->return_object(ptr);
            }
            return -1;
        }

        void clear_objects() {
            LocalPool *lp = _local_pool;
            if (lp) {
                _local_pool = nullptr;
                turbo::thread_atexit_cancel(LocalPool::delete_local_pool, lp);
                delete lp;
            }
        }

        inline static size_t free_chunk_nitem() {
            const size_t n = ObjectPoolTraits<T>::free_chunk_max_items();
            return (n < FREE_CHUNK_NITEM ? n : FREE_CHUNK_NITEM);
        }

        // Number of all allocated objects, including being used and free.
        [[nodiscard]] ObjectPoolInfo describe_objects() const {
            ObjectPoolInfo info;
            info.local_pool_num = _nlocal.load(std::memory_order_relaxed);
            info.block_group_num = _ngroup.load(std::memory_order_acquire);
            info.block_num = 0;
            info.item_num = 0;
            info.free_chunk_item_num = free_chunk_nitem();
            info.block_item_num = BLOCK_NITEM;
            info.free_item_num = _global_nfree.load(std::memory_order_relaxed);

            for (size_t i = 0; i < info.block_group_num; ++i) {
                BlockGroup *bg = _block_groups[i].load(std::memory_order_consume);
                if (nullptr == bg) {
                    break;
                }
                size_t nblock = std::min(bg->nblock.load(std::memory_order_relaxed),
                                         OP_GROUP_NBLOCK);
                info.block_num += nblock;
                for (size_t j = 0; j < nblock; ++j) {
                    Block *b = bg->blocks[j].load(std::memory_order_consume);
                    if (nullptr != b) {
                        info.item_num += b->nitem;
                    }
                }
            }
            info.total_size = info.block_num * info.block_item_num * sizeof(T);
            return info;
        }

        static inline ObjectPool *singleton() {
            ObjectPool *p = _singleton.load(std::memory_order_consume);
            if (p) {
                return p;
            }
            std::unique_lock l(_singleton_mutex);
            p = _singleton.load(std::memory_order_consume);
            if (!p) {
                p = new ObjectPool();
                _singleton.store(p, std::memory_order_release);
            }
            return p;
        }

    private:
        ObjectPool() {
            _free_chunks.reserve(OP_INITIAL_FREE_LIST_SIZE);
        }

        ~ObjectPool() = default;

        // Create a Block and append it to right-most BlockGroup.
        static Block *add_block(size_t *index) {
            auto *const new_block = new(std::nothrow) Block;
            if (nullptr == new_block) {
                return nullptr;
            }
            size_t ngroup;
            do {
                ngroup = _ngroup.load(std::memory_order_acquire);
                if (ngroup >= 1) {
                    BlockGroup *const g =
                            _block_groups[ngroup - 1].load(std::memory_order_consume);
                    const size_t block_index =
                            g->nblock.fetch_add(1, std::memory_order_relaxed);
                    if (block_index < OP_GROUP_NBLOCK) {
                        g->blocks[block_index].store(
                                new_block, std::memory_order_release);
                        *index = (ngroup - 1) * OP_GROUP_NBLOCK + block_index;
                        return new_block;
                    }
                    g->nblock.fetch_sub(1, std::memory_order_relaxed);
                }
            } while (add_block_group(ngroup));

            // Fail to add_block_group.
            delete new_block;
            return nullptr;
        }

        // Create a BlockGroup and append it to _block_groups.
        // Shall be called infrequently because a BlockGroup is pretty big.
        static bool add_block_group(size_t old_ngroup) {
            BlockGroup *bg = nullptr;
            std::unique_lock l(_block_group_mutex);
            const size_t ngroup = _ngroup.load(std::memory_order_acquire);
            if (ngroup != old_ngroup) {
                // Other thread got lock and added group before this thread.
                return true;
            }
            if (ngroup < OP_MAX_BLOCK_NGROUP) {
                bg = new(std::nothrow) BlockGroup;
                if (nullptr != bg) {
                    // Release fence is paired with consume fence in add_block()
                    // to avoid un-constructed bg to be seen by other threads.
                    _block_groups[ngroup].store(bg, std::memory_order_release);
                    _ngroup.store(ngroup + 1, std::memory_order_release);
                }
            }
            return bg != nullptr;
        }

        inline LocalPool *get_or_new_local_pool() {
            LocalPool *lp = _local_pool;
            if (TURBO_LIKELY(lp != nullptr)) {
                return lp;
            }
            lp = new(std::nothrow) LocalPool(this);
            if (nullptr == lp) {
                return nullptr;
            }
            std::unique_lock l(_change_thread_mutex); //avoid race with clear()
            _local_pool = lp;
            turbo::thread_atexit(LocalPool::delete_local_pool, lp);
            _nlocal.fetch_add(1, std::memory_order_relaxed);
            return lp;
        }

        void clear_from_destructor_of_local_pool() {
            // Remove tls
            _local_pool = nullptr;

            // Do nothing if there're active threads.
            if (_nlocal.fetch_sub(1, std::memory_order_relaxed) != 1) {
                return;
            }

            // Can't delete global even if all threads(called ObjectPool
            // functions) quit because the memory may still be referenced by
            // other threads. But we need to validate that all memory can
            // be deallocated correctly in tests, so wrap the function with
            // a macro which is only defined in unittests.
#ifdef TURBO_MEMORY_OBJECT_POOL_TEST
            std::unique_lock l(_change_thread_mutex);  // including acquire fence.
            // Do nothing if there're active threads.
            if (_nlocal.load(std::memory_order_relaxed) != 0) {
                return;
            }
            // All threads exited and we're holding _change_thread_mutex to avoid
            // racing with new threads calling get_object().

            // Clear global free list.
            FreeChunk dummy;
            while (pop_free_chunk(dummy));

            // Delete all memory
            const size_t ngroup = _ngroup.exchange(0, std::memory_order_relaxed);
            for (size_t i = 0; i < ngroup; ++i) {
                BlockGroup *bg = _block_groups[i].load(std::memory_order_relaxed);
                if (nullptr == bg) {
                    break;
                }
                size_t nblock = std::min(bg->nblock.load(std::memory_order_relaxed),
                                         OP_GROUP_NBLOCK);
                for (size_t j = 0; j < nblock; ++j) {
                    Block *b = bg->blocks[j].load(std::memory_order_relaxed);
                    if (nullptr == b) {
                        continue;
                    }
                    for (size_t k = 0; k < b->nitem; ++k) {
                        T *const objs = (T *) b->items;
                        objs[k].~T();
                    }
                    delete b;
                }
                delete bg;
            }

            memset(_block_groups, 0, sizeof(BlockGroup *) * OP_MAX_BLOCK_NGROUP);
#endif
        }

    private:
        bool pop_free_chunk(FreeChunk &c) {
            // Critical for the case that most return_object are called in
            // different threads of get_object.
            if (_free_chunks.empty()) {
                return false;
            }
            _free_chunks_mutex.lock();
            if (_free_chunks.empty()) {
                _free_chunks_mutex.unlock();
                return false;
            }
            DynamicFreeChunk *p = _free_chunks.back();
            _free_chunks.pop_back();
            _free_chunks_mutex.unlock();
            c.nfree = p->nfree;
            memcpy(c.ptrs, p->ptrs, sizeof(*p->ptrs) * p->nfree);
            free(p);
            return true;
        }

        bool push_free_chunk(const FreeChunk &c) {
            auto *p = (DynamicFreeChunk *) malloc(
                    offsetof(DynamicFreeChunk, ptrs) + sizeof(*c.ptrs) * c.nfree);
            if (!p) {
                return false;
            }
            p->nfree = c.nfree;
            memcpy(p->ptrs, c.ptrs, sizeof(*c.ptrs) * c.nfree);
            _free_chunks_mutex.lock();
            _free_chunks.push_back(p);
           _free_chunks_mutex.unlock();
            return true;
        }

        static std::atomic<ObjectPool *> _singleton;
        static std::mutex _singleton_mutex;
        static __thread LocalPool *_local_pool;
        static std::atomic<long> _nlocal;
        static std::atomic<size_t> _ngroup;
        static std::mutex _block_group_mutex;
        static std::mutex _change_thread_mutex;
        static std::atomic<BlockGroup *> _block_groups[OP_MAX_BLOCK_NGROUP];

        std::vector<DynamicFreeChunk *> _free_chunks;
        std::mutex _free_chunks_mutex;

        static std::atomic<size_t> _global_nfree;
    };

    template<typename T>
    std::atomic<ObjectPool<T> *> ObjectPool<T>::_singleton{nullptr};

    template<typename T>
    std::mutex ObjectPool<T>::_singleton_mutex;

    template<typename T>
    __thread typename ObjectPool<T>::LocalPool *ObjectPool<T>::_local_pool{nullptr};

    template<typename T>
    std::atomic<long> ObjectPool<T>::_nlocal{0};

    template<typename T>
    std::atomic<size_t> ObjectPool<T>::_ngroup{0};

    template<typename T>
    std::mutex ObjectPool<T>::_block_group_mutex;

    template<typename T>
    std::mutex ObjectPool<T>::_change_thread_mutex;

    template<typename T>
    std::atomic<size_t> ObjectPool<T>::_global_nfree;

    template<typename T>
    std::atomic<typename ObjectPool<T>::BlockGroup *>
            ObjectPool<T>::_block_groups[OP_MAX_BLOCK_NGROUP] = {};

    inline std::ostream &operator<<(std::ostream &os,
                                    ObjectPoolInfo const &info) {
        return os << "local_pool_num: " << info.local_pool_num
                  << "\nblock_group_num: " << info.block_group_num
                  << "\nblock_num: " << info.block_num
                  << "\nitem_num: " << info.item_num
                  << "\nblock_item_num: " << info.block_item_num
                  << "\nfree_chunk_item_num: " << info.free_chunk_item_num
                  << "\ntotal_size: " << info.total_size
                  << "\nfree_num: " << info.free_item_num;
    }
}  // namespace turbo

#endif  // TURBO_MEMORY_OBJECT_POOL_IMPL_H_
