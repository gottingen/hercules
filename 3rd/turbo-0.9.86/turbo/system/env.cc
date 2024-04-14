// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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

#include "turbo/system/env.h"

namespace turbo {

    std::string get_env(const std::string &str) {
#ifdef _MSC_VER
        char *ptr = nullptr;
  size_t len = 0;

  if(_dupenv_s(&ptr, &len, str.c_str()) == 0 && ptr != nullptr) {
    std::string res(ptr, len);
    std::free(ptr);
    return res;
  }
  return "";

#else
        auto ptr = std::getenv(str.c_str());
        return ptr ? ptr : "";
#endif
    }

    bool has_env(const std::string &str) {
#ifdef _MSC_VER
        char *ptr = nullptr;
  size_t len = 0;

  if(_dupenv_s(&ptr, &len, str.c_str()) == 0 && ptr != nullptr) {
    std::string res(ptr, len);
    std::free(ptr);
    return true;
  }
  return false;

#else
        auto ptr = std::getenv(str.c_str());
        return ptr ? true : false;
#endif
    }

    namespace {

#ifdef _WIN32
        const int kMaxEnvVarValueSize = 1024;
#endif

        void SetEnvVar(const char *name, const char *value) {
#ifdef _WIN32
            SetEnvironmentVariableA(name, value);
#else
            if (value == nullptr) {
                ::unsetenv(name);
            } else {
                ::setenv(name, value, 1);
            }
#endif
        }

    }  // namespace

    ScopedSetEnv::ScopedSetEnv(const char *var_name, const char *new_value)
            : var_name_(var_name), was_unset_(false) {
#ifdef _WIN32
        char buf[kMaxEnvVarValueSize];
            auto get_res = GetEnvironmentVariableA(var_name_.c_str(), buf, sizeof(buf));
            TURBO_INTERNAL_CHECK(get_res < sizeof(buf), "value exceeds buffer size");

            if (get_res == 0) {
              was_unset_ = (GetLastError() == ERROR_ENVVAR_NOT_FOUND);
            } else {
              old_value_.assign(buf, get_res);
            }

            SetEnvironmentVariableA(var_name_.c_str(), new_value);
#else
        const char *val = ::getenv(var_name_.c_str());
        if (val == nullptr) {
            was_unset_ = true;
        } else {
            old_value_ = val;
        }
#endif

        SetEnvVar(var_name_.c_str(), new_value);
    }

    ScopedSetEnv::~ScopedSetEnv() {
        SetEnvVar(var_name_.c_str(), was_unset_ ? nullptr : old_value_.c_str());
    }

}  // namespace turbo

