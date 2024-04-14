//
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

#include "stack_consumption.h"

#ifdef TURBO_INTERNAL_HAVE_DEBUGGING_STACK_CONSUMPTION

#include <string.h>

#include "gtest/gtest.h"
#include "turbo/base/internal/raw_logging.h"

namespace turbo::debugging_internal {
    namespace {

        static void SimpleSignalHandler(int signo) {
            char buf[100];
            memset(buf, 'a', sizeof(buf));

            // Never true, but prevents compiler from optimizing buf out.
            if (signo == 0) {
                TURBO_RAW_LOG(INFO, "%p", static_cast<void *>(buf));
            }
        }

        TEST(SignalHandlerStackConsumptionTest, MeasuresStackConsumption) {
            // Our handler should consume reasonable number of bytes.
            EXPECT_GE(GetSignalHandlerStackConsumption(SimpleSignalHandler), 100);
        }

    }  // namespace
}  // namespace turbo::debugging_internal

#endif  // TURBO_INTERNAL_HAVE_DEBUGGING_STACK_CONSUMPTION
