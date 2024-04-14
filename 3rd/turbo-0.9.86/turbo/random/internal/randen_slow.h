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

#ifndef TURBO_RANDOM_INTERNAL_RANDEN_SLOW_H_
#define TURBO_RANDOM_INTERNAL_RANDEN_SLOW_H_

#include <cstddef>

#include "turbo/platform/port.h"

namespace turbo::random_internal {

    // RANDen = RANDom generator or beetroots in Swiss High German.
    // RandenSlow implements the basic state manipulation methods for
    // architectures lacking AES hardware acceleration intrinsics.
    class RandenSlow {
    public:
        static void Generate(const void *keys, void *state_void);

        static void Absorb(const void *seed_void, void *state_void);

        static const void *GetKeys();
    };

}  // namespace turbo::random_internal

#endif  // TURBO_RANDOM_INTERNAL_RANDEN_SLOW_H_
