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

#ifndef TESTS_FLAGS_TEST_HELPER_H_
#define TESTS_FLAGS_TEST_HELPER_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "turbo/flags/flags.h"

using input_t = std::vector<std::string>;

class TApp {
public:
    turbo::App app{"My Test Program"};
    input_t args{};

    virtual ~TApp() = default;

    void run() {
        // It is okay to re-parse - clear is called automatically before a parse.
        input_t newargs = args;
        std::reverse(std::begin(newargs), std::end(newargs));
        app.parse(newargs);
    }
};

inline int fileClear(const std::string &name) { return std::remove(name.c_str()); }

class TempFile {
    std::string _name{};

public:
    explicit TempFile(std::string name) : _name(std::move(name)) {
        if (!turbo::NonexistentPath(_name).empty())
            throw std::runtime_error(_name);
    }

    ~TempFile() {
        std::remove(_name.c_str());  // Doesn't matter if returns 0 or not
    }

    operator const std::string &() const { return _name; }  // NOLINT(google-explicit-constructor)
    [[nodiscard]] const char *c_str() const { return _name.c_str(); }
};

inline void put_env(std::string name, std::string value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

inline void unset_env(std::string name) {
#ifdef _WIN32
    _putenv_s(name.c_str(), "");
#else
    unsetenv(name.c_str());
#endif
}

# endif  // TESTS_FLAGS_TEST_HELPER_H_
