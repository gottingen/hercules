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

#ifndef TURBO_STATUS_ERRNO_H_
#define TURBO_STATUS_ERRNO_H_


#include "turbo/platform/port.h"
#include <string>
#include <string_view>
#include <array>
#include <system_error>

//-----------------------------------------
// Use system errno before defining yours !
//-----------------------------------------
//
// To add new errno, you shall define the errno in header first, either by
// macro or constant, or even in protobuf.
//
//     #define ESTOP -114                // C/C++
//     static const int EMYERROR = 30;   // C/C++
//     const int EMYERROR2 = -31;        // C++ only
//
// Then you can register description of the error by calling
// TURBO_REGISTER_ERRNO(the_error_number, its_description) in global scope of
// a .cc or .cc files which will be linked.
// 
//     TURBO_REGISTER_ERRNO(ESTOP, "the thread is stopping")
//     TURBO_REGISTER_ERRNO(EMYERROR, "my error")
//
// Once the error is successfully defined:
//     turbo_error(error_code) returns the description.
//     turbo_error() returns description of last system error code.
//
// %m in printf-alike functions does NOT recognize errors defined by
// TURBO_REGISTER_ERRNO, you have to explicitly print them by %s.
//
//     errno = ESTOP;
//     printf("Something got wrong, %m\n");            // NO
//     printf("Something got wrong, %s\n", turbo_error());  // YES
//
// When the error number is re-defined, a linking error will be reported:
// 
//     "redefinition of `class TurboErrorHelper<30>'"
//
// Or the program aborts at runtime before entering main():
// 
//     "Fail to define EMYERROR(30) which is already defined as `Read-only file system', abort"
//

namespace turbo {
    using StatusCode = int;

    static constexpr StatusCode ERRNO_BEGIN = -32768;

    static constexpr StatusCode ERRNO_END = 32768;

    extern std::array<const char*, ERRNO_END - ERRNO_BEGIN> errno_desc_array;

    // You should not call this function, use TURBO_REGISTER_ERRNO instead.
    extern int describe_customized_errno(int, const char *, const char *);

    template<int error_code>
    class TurboErrorHelper {
    };
}  // namespace turbo::base

template<int error_code>
class TurboErrorRegister {
};

#define TURBO_REGISTER_ERRNO(error_code, description)                   \
    const int TURBO_MAYBE_UNUSED TURBO_CONCAT(turbo_errno_dummy_, __LINE__) =              \
        ::turbo::describe_customized_errno((error_code), #error_code, (description));        \
    template <> class TurboErrorRegister<(int)(error_code)> {};

#define TURBO_DECLARE_ERRNO(name, error_code) \
    static constexpr int name = error_code;    \
    template <> class TurboErrorHelper<(int)(error_code)> {};


#ifndef TURBO_PLATFORM_WINDOWS
namespace turbo {
    constexpr const char* terror(int error_code) {
        if (error_code >= turbo::ERRNO_BEGIN && error_code < turbo::ERRNO_END) {
            return turbo::errno_desc_array[error_code - turbo::ERRNO_BEGIN];
        }
        return "";
    }

    inline const char* terror() {
        return terror(errno);
    }

    inline std::error_code make_system_error(int err = 0) {
        return std::error_code(err ? err : errno, std::system_category());
    }

}  // namespace turbo
#endif  // TURBO_PLATFORM_WINDOWS
#ifdef TURBO_PLATFORM_WINDOWS
namespace turbo {

    inline std::error_code make_system_error(uint32_t err) {
        return std::error_code(err ? static_cast<int>(err)
                             : static_cast<int>(::GetLastError()),
                            std::system_category());
    }

}  // namespace turbo
#endif  // TURBO_PLATFORM_WINDOWS
#endif  // TURBO_STATUS_ERRNO_H_
