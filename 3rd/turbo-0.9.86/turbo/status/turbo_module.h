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
#ifndef TURBO_STATUS_TURBO_MODULE_H_
#define TURBO_STATUS_TURBO_MODULE_H_

#include "turbo/platform/port.h"
#include <errno.h> // errno

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
//     turbo_error(module_index) returns the description.
//     turbo_error() returns description of last system error code.
//
// %m in printf-alike functions does NOT recognize errors defined by
// TURBO_REGISTER_ERRNO, you have to explicitly print them by %s.
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
    // You should not call this function, use TURBO_REGISTER_ERRNO instead.
    extern int DescribeCustomizedModule(int, const char *, const char *);
}  // namespace turbo::base

template<unsigned short int module_index>
class TurboModuleHelper {
};

#define TURBO_REGISTER_MODULE_INDEX(module_index, description)                   \
    const int TURBO_MAYBE_UNUSED TURBO_CONCAT(turbo_module_dummy_, __LINE__) =              \
        ::turbo::DescribeCustomizedModule((module_index), #module_index, (description)); \
    template <> class TurboModuleHelper<(unsigned short int)(module_index)> {};

const char *TurboModule(int module_index);

namespace turbo {
    static constexpr unsigned short int kTurboModuleIndex = 0;
}  // namespace turbo

#endif  // TURBO_STATUS_TURBO_MODULE_H_
