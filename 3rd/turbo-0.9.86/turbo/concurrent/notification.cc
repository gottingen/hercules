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
#include "turbo/concurrent/notification.h"

namespace turbo {

    Notification::Notification(): _data(std::make_shared<inner_data>()) {
    }
    Notification::Notification(bool count) : _data(std::make_shared<inner_data>()) {
        _data->count = count;
    }

    void Notification::Notify() {
        _data->count = true;
        std::unique_lock lk(_data->mutex);
        _data->cond.notify_all();
    }

    bool Notification::HasBeenNotified() const noexcept {
        std::unique_lock lk(_data->mutex);
        return _data->count;
    }

    void Notification::WaitForNotification() const {
        std::unique_lock lk(_data->mutex);
        return _data->cond.wait(lk, [this] { return _data->count == true; });
    }

}  // namespace turbo