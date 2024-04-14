// Copyright 2020 The Turbo Authors.
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

#include "turbo/random/seed_gen_exception.h"

#include <iostream>

#include "turbo/platform/port.h"

namespace turbo {

    static constexpr const char kExceptionMessage[] =
            "Failed generating seed-material for URBG.";

    SeedGenException::~SeedGenException() = default;

    const char *SeedGenException::what() const noexcept {
        return kExceptionMessage;
    }

    namespace random_internal {

        void ThrowSeedGenException() {
#ifdef TURBO_HAVE_EXCEPTIONS
            throw turbo::SeedGenException();
#else
            std::cerr << kExceptionMessage << std::endl;
            std::terminate();
#endif
        }

    }  // namespace random_internal
}  // namespace turbo
