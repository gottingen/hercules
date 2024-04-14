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

#include "turbo/concurrent/hash_lock.h"
int main() {

    turbo::HashLock<size_t> hlock;
    {
        turbo::SharedLockGuard guard(hlock, 42ul);
    }

    {
        turbo::LockGuard guard(hlock, 42ul);
    }
    /* this will not work by type = std::strrng
    {
        std::vector<std::string> ks{"1", "3"};
        auto ls = hlock.multi_get(ks);
        turbo::SharedLockGuard<size_t> guard(&hlock, 42);
    }
     */
    {
        std::vector<size_t> ks{1,2};
        auto ls = hlock.multi_get(ks);
        turbo::MultiSharedLockGuard<size_t> guard(&hlock, ks);
    }

    {
        std::set<size_t> ks{1,2};
        auto ls = hlock.multi_get(ks);
        turbo::MultiLockGuard<size_t> guard(&hlock, ks);
    }
}