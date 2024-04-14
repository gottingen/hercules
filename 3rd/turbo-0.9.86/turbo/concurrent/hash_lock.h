// Copyright 2023 The Turbo Authors.
//
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

#ifndef TURBO_CONCURRENT_HASH_LOCK_H_
#define TURBO_CONCURRENT_HASH_LOCK_H_

#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <thread>
#include <string_view>
#include "turbo/platform/port.h"
#include "turbo/meta/span.h"
#include "turbo/meta/type_traits.h"


namespace turbo {


    template <typename T>
    class HashLock {
    public:
        static constexpr int kDefaultHashPower = 16;
        HashLock() :_hash_mask((1U << _hash_power) - 1) {
            for (unsigned i = 0; i < size(); i++) {
                _mutex_pool.emplace_back(new std::shared_mutex{});
            }
        }

        explicit HashLock(int hash_power): _hash_power(hash_power),_hash_mask((1U << _hash_power) - 1) {
            for (unsigned i = 0; i < size(); i++) {
                _mutex_pool.emplace_back(new std::shared_mutex{});
            }
        }
        ~HashLock() = default;

        HashLock(const HashLock &) = delete;

        HashLock &operator=(const HashLock &) = delete;

        unsigned size() const {
            return (1U << _hash_power);
        }

        std::shared_mutex* get_lock(const T &key) const {
            return  _mutex_pool[Hash_impl(key)].get();
        }

        template<typename C, TURBO_REQUIRES(std::is_same<typename C::value_type, T>)>
        std::vector<std::shared_mutex *> multi_get(const C &keys) const {
            std::set<unsigned, std::greater<unsigned>> to_acquire_indexes;
            // We are using the `set` to avoid retrieving the mutex, as well as guarantee to retrieve
            // the order of locks.
            //
            // For example, we need lock the key `A` and `B` and they have the same lock Hash_impl
            // index, it will be deadlock if lock the same mutex twice. Besides, we also need
            // to order the mutex before acquiring locks since different threads may acquire
            // same keys with different order.
            for (const auto &key: keys) {
                to_acquire_indexes.insert(Hash_impl(key));
            }

            std::vector<std::shared_mutex *> locks;
            locks.reserve(to_acquire_indexes.size());
            for (auto index: to_acquire_indexes) {
                locks.emplace_back(_mutex_pool[index].get());
            }
            return locks;
        }

    private:
        int _hash_power{kDefaultHashPower};
        unsigned _hash_mask;
        std::vector<std::unique_ptr<std::shared_mutex>> _mutex_pool;

        unsigned Hash_impl(const T &key) const {
            return std::hash<T>{}(key) & _hash_mask;
        }
    };

    class SharedLockGuard {
    public:
        template<typename T>
        explicit SharedLockGuard(HashLock<T> &lock_mgr, const T &key) : _smutex(lock_mgr.get_lock(key)) {
            _smutex->lock_shared();
        }

        ~SharedLockGuard() { _smutex->unlock_shared(); }

        SharedLockGuard(const SharedLockGuard &) = delete;

        SharedLockGuard &operator=(const SharedLockGuard &) = delete;

    private:
        std::shared_mutex *_smutex{nullptr};
    };

    class LockGuard {
    public:
        template<typename T>
        explicit LockGuard(HashLock<T> &lock_mgr,const T &key) : _smutex(lock_mgr.get_lock(key)){
            _smutex->lock();
        }

        ~LockGuard() { _smutex->unlock(); }

        LockGuard(const LockGuard &) = delete;

        LockGuard &operator=(const LockGuard &) = delete;

    private:
        std::shared_mutex *_smutex{nullptr};
    };

    template<typename T>
    class MultiSharedLockGuard {
    public:
        template<typename C, TURBO_REQUIRES(std::is_same<typename C::value_type, T>)>
        explicit MultiSharedLockGuard(HashLock<T> *lock_mgr, const C &keys) : lock_mgr_(lock_mgr) {
            locks_ = lock_mgr_->multi_get(keys);
            for (const auto &iter: locks_) {
                iter->lock_shared();
            }
        }

        ~MultiSharedLockGuard() {
            for (auto iter = locks_.rbegin(); iter != locks_.rend(); ++iter) {
                (*iter)->unlock_shared();
            }
        }

        MultiSharedLockGuard(const MultiSharedLockGuard &) = delete;

        MultiSharedLockGuard &operator=(const MultiSharedLockGuard &) = delete;

    private:
        HashLock<T> *lock_mgr_ = nullptr;
        std::vector<std::shared_mutex *> locks_;
    };

    template<typename T>
    class MultiLockGuard {
    public:
        template<typename C, TURBO_REQUIRES(std::is_same<typename C::value_type, T>)>
        explicit MultiLockGuard(HashLock<T> *lock_mgr, const C &keys) : lock_mgr_(lock_mgr) {
            locks_ = lock_mgr_->multi_get(keys);
            for (const auto &iter: locks_) {
                iter->lock();
            }
        }

        ~MultiLockGuard() {
            for (auto iter = locks_.rbegin(); iter != locks_.rend(); ++iter) {
                (*iter)->unlock();
            }
        }

        MultiLockGuard(const MultiLockGuard &) = delete;

        MultiLockGuard &operator=(const MultiLockGuard &) = delete;

    private:
        HashLock<T> *lock_mgr_ = nullptr;
        std::vector<std::shared_mutex *> locks_;
    };


}
#endif  // TURBO_CONCURRENT_HASH_LOCK_H_
