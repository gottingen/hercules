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
#ifndef TURBO_CONTAINER_ARRAY_VIEW_H_
#define TURBO_CONTAINER_ARRAY_VIEW_H_

#include <cstdint>
#include <memory>
#include <cstddef>

namespace turbo {

    template<typename T>
    class array_view {
    public:
        array_view();

        array_view(T *p_array, std::size_t p_length, bool p_transferOwnership);

        array_view(T *p_array, std::size_t p_length, std::shared_ptr<T> p_dataHolder);

        array_view(array_view<T> &&p_right);

        array_view(const array_view<T> &p_right);

        array_view<T> &operator=(array_view<T> &&p_right);

        array_view<T> &operator=(const array_view<T> &p_right);

        T &operator[](std::size_t p_index);

        const T &operator[](std::size_t p_index) const;

        ~array_view();

        T *data() const;

        std::size_t size() const;

        std::shared_ptr<T> data_holder() const;

        void set(T *p_array, std::size_t p_length, bool p_transferOwnership);

        void clear();

        static array_view<T> alloc(std::size_t p_length);

        const static array_view<T> c_empty;

    private:
        T *_data;

        std::size_t _length;

        // Notice this is holding an array. Set correct deleter for this.
        std::shared_ptr<T> _dataHolder;
    };

    template<typename T>
    const array_view<T> array_view<T>::c_empty;


    template<typename T>
    array_view<T>::array_view()
            : _data(nullptr),
              _length(0) {
    }

    template<typename T>
    array_view<T>::array_view(T *p_array, std::size_t p_length, bool p_transferOnwership)

            : _data(p_array),
              _length(p_length) {
        if (p_transferOnwership) {
            _dataHolder.reset(_data, std::default_delete<T[]>());
        }
    }


    template<typename T>
    array_view<T>::array_view(T *p_array, std::size_t p_length, std::shared_ptr<T> p_dataHolder)
            : _data(p_array),
              _length(p_length),
              _dataHolder(std::move(p_dataHolder)) {
    }


    template<typename T>
    array_view<T>::array_view(array_view<T> &&p_right)
            : _data(p_right._data),
              _length(p_right._length),
              _dataHolder(std::move(p_right._dataHolder)) {
    }


    template<typename T>
    array_view<T>::array_view(const array_view<T> &p_right)
            : _data(p_right._data),
              _length(p_right._length),
              _dataHolder(p_right._dataHolder) {
    }


    template<typename T>
    array_view<T> &
    array_view<T>::operator=(array_view<T> &&p_right) {
        _data = p_right._data;
        _length = p_right._length;
        _dataHolder = std::move(p_right._dataHolder);

        return *this;
    }


    template<typename T>
    array_view<T> &
    array_view<T>::operator=(const array_view<T> &p_right) {
        _data = p_right._data;
        _length = p_right._length;
        _dataHolder = p_right._dataHolder;

        return *this;
    }


    template<typename T>
    T &
    array_view<T>::operator[](std::size_t p_index) {
        return _data[p_index];
    }


    template<typename T>
    const T &
    array_view<T>::operator[](std::size_t p_index) const {
        return _data[p_index];
    }


    template<typename T>
    array_view<T>::~array_view() {
    }


    template<typename T>
    T *
    array_view<T>::data() const {
        return _data;
    }


    template<typename T>
    std::size_t
    array_view<T>::size() const {
        return _length;
    }


    template<typename T>
    std::shared_ptr<T>
    array_view<T>::data_holder() const {
        return _dataHolder;
    }


    template<typename T>
    void
    array_view<T>::set(T *p_array, std::size_t p_length, bool p_transferOwnership) {
        _data = p_array;
        _length = p_length;

        if (p_transferOwnership) {
            _dataHolder.reset(_data, std::default_delete<T[]>());
        }
    }


    template<typename T>
    void array_view<T>::clear() {
        _data = nullptr;
        _length = 0;
        _dataHolder.reset();
    }


    template<typename T>
    array_view<T>
    array_view<T>::alloc(std::size_t p_length) {
        array_view<T> arr;
        if (0 == p_length) {
            return arr;
        }

        arr._dataHolder.reset(new T[p_length], std::default_delete<T[]>());

        arr._length = p_length;
        arr._data = arr._dataHolder.get();
        return arr;
    }


    typedef array_view<std::uint8_t> byte_array_view;

}  // namespace turbo

#endif  // TURBO_CONTAINER_ARRAY_VIEW_H_

