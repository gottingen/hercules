// Copyright 2022 The Turbo Authors.
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

#ifndef TURBO_BASE_SINGLETON_H_
#define TURBO_BASE_SINGLETON_H_

#include <cstdlib>
#include <mutex>

namespace turbo {

template <class T>
class LeakedSingleton {
 public:
  friend T;

  static T* instance() {
    if (_s_instance == NULL) {
      std::unique_lock<std::mutex> lock(s_singleton_mutex_);
      if (_s_instance == NULL) {
        _s_instance = new T;
      }
    }
    return _s_instance;
  }

 private:
  static T* _s_instance;
  static std::mutex s_singleton_mutex_;
};

template <class T>
T* LeakedSingleton<T>::_s_instance = nullptr;

template <class T>
std::mutex LeakedSingleton<T>::s_singleton_mutex_;

}  // namespace turbo


#endif  // TURBO_BASE_SINGLETON_H_
