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
// Created by jeff on 24-1-12.
//

#ifndef TURBO_MEMORY_DEEP_COPY_PTR_H_
#define TURBO_MEMORY_DEEP_COPY_PTR_H_

namespace turbo {

    template<typename T>
    class DeepCopyPtr {
    public:
        DeepCopyPtr() : _ptr(nullptr) {}

        explicit DeepCopyPtr(T *obj) : _ptr(obj) {}

        ~DeepCopyPtr() {
            delete _ptr;
        }

        DeepCopyPtr(const DeepCopyPtr &rhs)
                : _ptr(rhs._ptr ? new T(*rhs._ptr) : nullptr) {}

        void operator=(const DeepCopyPtr &rhs) {
            if (this == &rhs) {
                return;
            }

            if (rhs._ptr) {
                if (_ptr) {
                    *_ptr = *rhs._ptr;
                } else {
                    _ptr = new T(*rhs._ptr);
                }
            } else {
                delete _ptr;
                _ptr = nullptr;
            }
        }

        T *get() const { return _ptr; }

        void reset(T *ptr) {
            delete _ptr;
            _ptr = ptr;
        }

        operator void *() const { return _ptr; }

        explicit operator bool() const { return get() != nullptr; }

        T &operator*() const { return *get(); }

        T *operator->() const { return get(); }

    private:
        T *_ptr;
    };

}  // namespace turbo

#endif  // TURBO_MEMORY_DEEP_COPY_PTR_H_
