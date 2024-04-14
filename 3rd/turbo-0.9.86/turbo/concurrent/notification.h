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
#ifndef TURBO_CONCURRENT_NOTIFICATION_H_
#define TURBO_CONCURRENT_NOTIFICATION_H_

#include <condition_variable>
#include <mutex>
#include <atomic>
#include <memory>
#include "turbo/times/time.h"
#include "turbo/log/logging.h"

namespace turbo {

    class Notification {
    public:
        Notification();
        explicit Notification(bool prenotify);
        Notification(const Notification&) = delete;
        Notification& operator=(const Notification&) = delete;

        // Notification::Notify()
        //
        // Sets the "notified" state of this notification to `true` and wakes waiting
        // threads. Note: do not call `Notify()` multiple times on the same
        // `Notification`; calling `Notify()` more than once on the same notification
        // results in undefined behavior.
        void Notify();

        // Notification::HasBeenNotified()
        //
        // Returns the value of the notification's internal "notified" state.
        [[nodiscard]] bool HasBeenNotified() const noexcept;

        // Notification::WaitForNotification()
        //
        // Blocks the calling thread until the notification's "notified" state is
        // `true`. Note that if `Notify()` has been previously called on this
        // notification, this function will immediately return.
        void WaitForNotification() const;

        // Notification::WaitForNotificationWithTimeout()
        //
        // Blocks until either the notification's "notified" state is `true` (which
        // may occur immediately) or the timeout has elapsed, returning the value of
        // its "notified" state in either case.
        bool WaitForNotificationWithTimeout(const turbo::Duration &d) const {
            std::chrono::microseconds timeout = d.to_chrono_microseconds();
            std::unique_lock lk(_data->mutex);
            TLOG_CHECK_GE(_data->count, 0ul);
            return _data->cond.wait_for(lk, timeout, [this] { return _data->count == 0; });
        }

        // Notification::WaitForNotificationWithDeadline()
        //
        // Blocks until either the notification's "notified" state is `true` (which
        // may occur immediately) or the deadline has expired, returning the value of
        // its "notified" state in either case.
        bool WaitForNotificationWithDeadline(const turbo::Time &deadline) const {
            auto d = deadline.to_chrono_time();
            std::unique_lock lk(_data->mutex);
            TLOG_CHECK_GE(_data->count, 0ul);
            return _data->cond.wait_until(lk, d, [this] { return _data->count == 0; });
        }

    private:
        struct inner_data {
            std::mutex mutex;
            std::condition_variable cond;
            std::atomic<bool> count{false};
        };
        std::shared_ptr<inner_data> _data;
    };

}  // namespace turbo
#endif  // TURBO_CONCURRENT_NOTIFICATION_H_
