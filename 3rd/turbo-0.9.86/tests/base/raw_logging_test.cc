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

// This test serves primarily as a compilation test for base/raw_logging.h.
// Raw logging testing is covered by logging_unittest.cc, which is not as
// portable as this test.

#include "turbo/base/internal/raw_logging.h"

#include <tuple>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"
#include "turbo/format/format.h"

namespace {

    TEST_CASE("RawLoggingCompilationTest, Log") {
        TURBO_RAW_LOG(INFO, "RAW INFO: %d", 1);
        TURBO_RAW_LOG(INFO, "RAW INFO: %d %d", 1, 2);
        TURBO_RAW_LOG(INFO, "RAW INFO: %d %d %d", 1, 2, 3);
        TURBO_RAW_LOG(INFO, "RAW INFO: %d %d %d %d", 1, 2, 3, 4);
        TURBO_RAW_LOG(INFO, "RAW INFO: %d %d %d %d %d", 1, 2, 3, 4, 5);
        TURBO_RAW_LOG(WARNING, "RAW WARNING: %d", 1);
        TURBO_RAW_LOG(ERROR, "RAW ERROR: %d", 1);
    }

    TEST_CASE("RawLoggingCompilationTest, PassingCheck") {
        TURBO_RAW_CHECK(true, "RAW CHECK");
    }


    TEST_CASE("InternalLog, CompilationTest") {
        TURBO_INTERNAL_LOG(INFO, "Internal Log");
        std::string log_msg = "Internal Log";
        TURBO_INTERNAL_LOG(INFO, log_msg);

        TURBO_INTERNAL_LOG(INFO, log_msg + " 2");

        float d = 1.1f;
        TURBO_INTERNAL_LOG(INFO, turbo::format("Internal log {} + {}", 3, d));
    }

}  // namespace
