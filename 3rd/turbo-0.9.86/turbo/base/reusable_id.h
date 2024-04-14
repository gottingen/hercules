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
// Created by jeff on 23-12-16.
//

#ifndef TURBO_BASE_REUSABLE_ID_H_
#define TURBO_BASE_REUSABLE_ID_H_

#include <deque>
#include <limits>
#include <cstdint>
#include <algorithm>
#include "turbo/status/result_status.h"

namespace turbo {

    /**
     * @ingroup turbo_base
     * @brief ReusableId is a class to generate reusable id, it is useful when we need to
     *        generate id for some object, and we want to reuse the id when the object is
     *        destroyed. For example, we have a pool of objects, and we want to reuse the
     *        id when the object is returned to the pool. The id is a unsigned integer,
     *        and the max id is configurable. The default max id is the max value of the
     *        id type. The most intuitive example is the file descriptor in Linux, it is
     *        a unsigned integer, and the max value is 2^32 - 1. When we close a file, the
     *        file descriptor is reusable.
     *        The ReusableId is not thread safe. It designed to be used some scenarios, user
     *        should use it in a context that should be guaranteed Thread race condition with
     *        some other variables together, it cost much using a lock to protect the id generation.
     *        For example, we have a pool of objects, and we want to reuse the id when the object
     *        is returned to the pool. the id generation operation is the the pool, when the pool
     *        is thread safe, the id generation is thread safe.
     *
     *        The Template parameter ``T'' is a helpful trait help you to make a global id singleton
     *        for a specific type. For example, we have a class ``Foo'', and we want to generate
     *        a global id for each instance of ``Foo'', we can use ``ReusableId<Foo>'' to generate,
     *        Example:
     *        @code
     *        class FooSingleton {
     *        private:
     *            static std::mutex _mutex;
     *        public:
     *            static uint32_t kMaxFooId = 4096;
     *            typedef ReusableId<Foo, uint32_t, kMaxFooId> FooIdType;
     *            static uint32_t get_id() {
     *                std::lock_guard<std::mutex> lock(_mutex);
     *                return FooIdType::get_instance().create_id().value_or(kInvalidId);
     *           }
     *           static free_id(uint32_t id) {
     *                 std::lock_guard<std::mutex> lock(_mutex);
     *                 FooIdType::get_instance().free_id(id);
     *                 FooIdType::get_instance().shrink_to_fit();
     *                 // shrink_to_fit is optional, it will reduce the memory usage
     *                 //  when the free ids ratio is too high
     *                 //  you can call it in a background thread
     *                 //  or in a context that is not performance sensitive
     *           }
     *           static clear() {
     *              std::lock_guard<std::mutex> lock(_mutex);
     *              FooIdType::get_instance().clear();
     *           }
     *       };
     *
     *           class Foo {
     *           public:
     *           Foo() : _id(FooSingleton::get_id()) {}
     *           ~Foo() {
     *              FooSingleton::free_id(_id);
     *           }
     *           bool is_valid() const {
     *                return _id !=
     *           private:
     *               uin32_t _id;
     *               // other members
     *            };
     * @note note that different max id is different type, like ``ReusableId<Foo, uint32_t, 4096>'' and
     *       ``ReusableId<Foo, uint32_t, 8192>'' is different type, when you call the singleton, you should
     *       know it. It is a good practice to define a typedef for the singleton.
     *
     */
    template<typename T, typename IdType = uint32_t, IdType MaxId = std::numeric_limits<IdType>::max()>
    class ReusableId {
    public:
        using id_type = IdType;
        using value_type = T;
        static constexpr IdType max_id = MaxId;
    public:
        static ReusableId &get_instance() {
            static ReusableId instance;
            return instance;
        }
        ReusableId() : _next_id(0) {}

        ReusableId(const ReusableId &) = delete;

        ReusableId &operator=(const ReusableId &) = delete;

        ReusableId(ReusableId &&) = default;

        ReusableId &operator=(ReusableId &&) = default;

        ~ReusableId() = default;

        [[nodiscard]] turbo::ResultStatus<IdType> create_id() {
            if (!_free_ids.empty()) {
                // pop front may case memory increase
                IdType id = _free_ids.front();
                _free_ids.pop_front();
                return id;
            } else if (_next_id < max_id) {
                return _next_id++;
            }
            return turbo::make_status(kEOVERFLOW, "ReusableId overflow");
        }

        void free_id(IdType id) {
            if (id == _next_id - 1) {
                --_next_id;
            } else {
                _free_ids.push_back(id);
            }
        }

        IdType next_id() const {
            return _next_id;
        }

        void clear() {
            _free_ids.clear();
            _next_id = 0;
        }

        size_t free_ids_size() const {
            return _free_ids.size();
        }

        double free_ids_ratio() const {
            return static_cast<double>(_free_ids.size()) / static_cast<double>(_next_id);
        }

        // for some condition, free ids ratio is too high, we need to shrink the free ids
        //  to reduce the memory usage

        id_type replacement(id_type id) const {
            free_id(id);
            // must be a valid id
            return create_id().value();
        }

        void shrink_to_fit() {
            std::deque<IdType> tmp = _free_ids;
            std::sort(tmp.begin(), tmp.end(), std::greater<IdType>());
            id_type  tmp_id = _next_id;
            auto it = tmp.begin();
            while (it != tmp.end()) {
                // avoid unsigned integer underflow
                if (*it == tmp_id - 1) {
                    --tmp_id;
                    ++it;
                } else {
                    break;
                }
            }

            if (it != tmp.begin()) {
                std::deque<IdType> smaller_ids;
                tmp.erase(tmp.begin(), it);
                tmp.shrink_to_fit();
                _free_ids.swap(tmp);
                _next_id = tmp_id;
            }
        }

    private:
        std::deque<IdType> _free_ids;
        IdType _next_id{0};
    };
}  // namespace turbo
#endif  // TURBO_BASE_REUSABLE_ID_H_
