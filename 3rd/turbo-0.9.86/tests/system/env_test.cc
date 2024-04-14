// Copyright 2019 The Turbo Authors.
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

#ifdef _WIN32
#include <windows.h>
#endif

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/system/env.h"

namespace {

    using turbo::ScopedSetEnv;

    std::string GetEnvVar(const char *name) {
#ifdef _WIN32
        char buf[1024];
        auto get_res = GetEnvironmentVariableA(name, buf, sizeof(buf));
        if (get_res >= sizeof(buf)) {
          return "TOO_BIG";
        }

        if (get_res == 0) {
          return "UNSET";
        }

        return std::string(buf, get_res);
#else
        const char *val = ::getenv(name);
        if (val == nullptr) {
            return "UNSET";
        }

        return val;
#endif
    }

    TEST_CASE("ScopedSetEnvTest, SetNonExistingVarToString") {
        CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "UNSET");

        {
            ScopedSetEnv scoped_set("SCOPED_SET_ENV_TEST_VAR", "value");

            CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "value");
        }

        CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "UNSET");
    }

    TEST_CASE("ScopedSetEnvTest, SetNonExistingVarToNull") {
        CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "UNSET");

        {
            ScopedSetEnv scoped_set("SCOPED_SET_ENV_TEST_VAR", nullptr);

            CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "UNSET");
        }

        CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "UNSET");
    }

    TEST_CASE("ScopedSetEnvTest, SetExistingVarToString") {
        ScopedSetEnv scoped_set("SCOPED_SET_ENV_TEST_VAR", "value");
        CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "value");

        {
            ScopedSetEnv scoped_set("SCOPED_SET_ENV_TEST_VAR", "new_value");

            CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "new_value");
        }

        CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "value");
    }

    TEST_CASE("ScopedSetEnvTest, SetExistingVarToNull") {
        ScopedSetEnv scoped_set("SCOPED_SET_ENV_TEST_VAR", "value");
        CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "value");

        {
            ScopedSetEnv scoped_set("SCOPED_SET_ENV_TEST_VAR", nullptr);

            CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "UNSET");
        }

        CHECK_EQ(GetEnvVar("SCOPED_SET_ENV_TEST_VAR"), "value");
    }

}  // namespace
