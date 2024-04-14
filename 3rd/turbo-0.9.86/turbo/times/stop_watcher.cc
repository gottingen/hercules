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
//

#include "turbo/times/stop_watcher.h"
#include <cassert>

namespace turbo {

    StopWatcher::StopWatcher(const std::string &title, time_print_t time_print)
    : _start(turbo::time_now()), _title(title),
    _time_print(std::move(time_print)),
    _duration(turbo::Duration::nanoseconds(0)) {
    }

    void StopWatcher::reset() {
        _start = turbo::time_now();
        _duration = turbo::Duration();
    }

    const StopWatcher &StopWatcher::stop() {
        auto n = turbo::time_now();
        if (_duration <= kZero) {
            _duration = n - _start;
        }
        return *this;
    }

    turbo::Duration StopWatcher::elapsed() const {
        if (_duration > kZero) {
            return  _duration;
        }
        return turbo::time_now() - _start;
    }
}  // namespace turbo
