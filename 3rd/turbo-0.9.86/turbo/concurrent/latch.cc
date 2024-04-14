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
#include "turbo/concurrent/latch.h"

namespace turbo {

    Latch::Latch(uint32_t count) : _data(std::make_shared<inner_data>()) {
        _data->count = count;
    }

    void Latch::CountDown(uint32_t update) {
        TLOG_CHECK(_data->count > 0, "turbo::Latch::CountDown() called too many times");
        _data->count -= update;
        if (_data->count == 0) {
            std::unique_lock lk(_data->mutex);
            _data->cond.notify_all();
        }
    }

    void Latch::CountUp(uint32_t update) {
        _data->count += update;
    }

    bool Latch::TryWait() const noexcept {
        std::unique_lock lk(_data->mutex);
        TLOG_CHECK_GE(_data->count, 0u);
        return !_data->count;
    }

    void Latch::Wait() const {
        std::unique_lock lk(_data->mutex);
        TLOG_CHECK_GE(_data->count, 0u);
        return _data->cond.wait(lk, [this] { return _data->count == 0; });
    }

    void Latch::ArriveAndWait(std::ptrdiff_t update) {
        CountDown(update);
        Wait();
    }

}  // namespace turbo