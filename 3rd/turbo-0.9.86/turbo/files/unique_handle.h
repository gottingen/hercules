// Copyright 2023 The titan-search Authors.
// by jeff.li
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

#ifndef TURBO_FILES_UNIQUE_HANDLE_H_
#define TURBO_FILES_UNIQUE_HANDLE_H_

#include <utility>

namespace turbo {
    
    template<typename T>
    class ResourceHandler {
    public:
        template<typename... Args>
        static int create_handle(T *handle, Args... args);

        static int destroy_handle(T handle);
    };

    /// \brief A generic class to manage basic RAII lifetimes for C handles
    ///
    /// This class manages the lifetimes of C handles found in many types of
    /// libraries. This class is non-copiable but can be moved.
    ///
    /// You can use this class with a new handle by using the DEFINE_HANDLER
    /// macro to define creatHandle/destroy_handle policy implemention for a
    /// given resource handle type.
    ///
    /// \code{.cpp}
    /// DEFINE_HANDLER(ClassName, HandleName, HandleCreator, HandleDestroyer);
    /// \code{.cpp}

    template<typename T>
    class unique_handle {
    private:
        T handle_;

    public:
        /// Default constructor. Initializes the handle to zero. Does not call the
        /// create function
        constexpr unique_handle() noexcept: handle_(0) {}

        /// \brief Takes ownership of a previously created handle
        ///
        /// \param[in] handle The handle to manage by this object
        explicit constexpr unique_handle(T handle) noexcept: handle_(handle) {};

        /// \brief Deletes the handle if created.
        ~unique_handle() noexcept { reset(); }

        /// \brief Deletes the handle if created.
        void reset() noexcept {
            if (handle_) {
                ResourceHandler<T>::destroy_handle(handle_);
                handle_ = 0;
            }
        }

        unique_handle(const unique_handle &other) noexcept = delete;

        unique_handle &operator=(unique_handle &other) noexcept = delete;

        constexpr unique_handle(unique_handle &&other) noexcept
                : handle_(other.handle_) {
            other.handle_ = 0;
        }

        unique_handle &operator=(unique_handle &&other) noexcept {
            handle_ = other.handle_;
            other.handle_ = 0;
        }

        /// \brief Implicit converter for the handle
        constexpr operator const T &() const noexcept { return handle_; }

        template<typename... Args>
        int create(Args... args) {
            if (!handle_) {
                int error = ResourceHandler<T>::create_handle(
                        &handle_, std::forward<Args>(args)...);
                if (error) { handle_ = 0; }
                return error;
            }
            return 0;
        }

        // Returns true if the \p other unique_handle is the same as this handle
        constexpr bool operator==(unique_handle &other) const noexcept {
            return handle_ == other.handle_;
        }

        // Returns true if the \p other handle is the same as this handle
        constexpr bool operator==(T &other) const noexcept {
            return handle_ == other;
        }

        // Returns true if the \p other handle is the same as this handle
        constexpr bool operator==(T other) const noexcept {
            return handle_ == other;
        }

        // Returns true if the handle was initialized correctly
        constexpr operator bool() { return handle_ != 0; }
    };

    /// \brief Returns an initialized handle object. The create function on this
    ///        object is already called with the parameter pack provided as
    ///        function arguments.
    template<typename T, typename... Args>
    unique_handle<T> make_handle(Args... args) {
        unique_handle<T> h;
        h.create(std::forward<Args>(args)...);
        return h;
    }
}  // namespace turbo

#define DEFINE_HANDLER(HANDLE_TYPE, HCREATOR, HDESTROYER)            \
    namespace turbo {                                            \
    template<>                                                       \
    class ResourceHandler<HANDLE_TYPE> {                             \
       public:                                                       \
        template<typename... Args>                                   \
        static int create_handle(HANDLE_TYPE *handle, Args... args) { \
            return HCREATOR(handle, std::forward<Args>(args)...);    \
        }                                                            \
        static int destroy_handle(HANDLE_TYPE handle) {               \
            return HDESTROYER(handle);                               \
        }                                                            \
    };                                                               \
    }

#endif  // TURBO_FILES_UNIQUE_HANDLE_H_
