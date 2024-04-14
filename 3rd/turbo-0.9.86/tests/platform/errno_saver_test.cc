// Copyright 2018 The Turbo Authors.
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

#include "turbo/platform/internal/errno_saver.h"

#include <cerrno>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/base/internal/strerror.h"

namespace {
    struct ErrnoPrinter {
        int no;
    };

    std::ostream &operator<<(std::ostream &os, ErrnoPrinter ep) {
        return os << turbo::base_internal::StrError(ep.no) << " [" << ep.no << "]";
    }

    bool operator==(ErrnoPrinter one, ErrnoPrinter two) { return one.no == two.no; }

    TEST_CASE("ErrnoSaverTest, Works") {
        errno = EDOM;
        {
            turbo::base_internal::ErrnoSaver errno_saver;
            CHECK_EQ(ErrnoPrinter{errno}, ErrnoPrinter{EDOM});
            errno = ERANGE;
            CHECK_EQ(ErrnoPrinter{errno}, ErrnoPrinter{ERANGE});
            CHECK_EQ(ErrnoPrinter{errno_saver()}, ErrnoPrinter{EDOM});
        }
        CHECK_EQ(ErrnoPrinter{errno}, ErrnoPrinter{EDOM});
    }
}  // namespace
