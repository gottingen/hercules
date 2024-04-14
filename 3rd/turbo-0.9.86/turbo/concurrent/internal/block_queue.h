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

#ifndef TURBO_CONCURRENT_INTERNAL_BLOCK_QUEUE_H_
#define TURBO_CONCURRENT_INTERNAL_BLOCK_QUEUE_H_


#include <condition_variable>
#include <mutex>
#include "turbo/container/ring_buffer.h"

namespace turbo::concurrent_internal{


    template<typename T>
    class BlockingQueue {
    public:
        using item_type = T;
        explicit BlockingQueue(size_t max_items)
                : q_(max_items)
        {}

        // try to enqueue and block if no room left
        void Enqueue(T &&item)
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                pop_cv_.wait(lock, [this] { return !this->q_.full(); });
                q_.push_back(std::move(item));
            }
            push_cv_.notify_one();
        }

        // enqueue immediately. overrun oldest message in the queue if no room left.
        void Enqueue_nowait(T &&item)
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                q_.push_back(std::move(item));
            }
            push_cv_.notify_one();
        }

        // try to dequeue item. if no item found. wait upto timeout and try again
        // Return true, if succeeded dequeue item, false otherwise
        bool DequeueFor(T &popped_item, std::chrono::milliseconds wait_duration)
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (!push_cv_.wait_for(lock, wait_duration, [this] { return !this->q_.empty(); }))
                {
                    return false;
                }
                popped_item = std::move(q_.front());
                q_.pop_front();
            }
            pop_cv_.notify_one();
            return true;
        }

        size_t OverrunCounter()
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            return q_.overrun_counter();
        }

    private:
        std::mutex queue_mutex_;
        std::condition_variable push_cv_;
        std::condition_variable pop_cv_;
        turbo::ring_buffer<T> q_;
    };

}  // namespace turbo::concurrent_internal

#endif  // TURBO_CONCURRENT_INTERNAL_BLOCK_QUEUE_H_
