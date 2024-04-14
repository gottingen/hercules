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
//#define CLOG_CHECK_ALWAYS_ON
#include <collie/log/logging.h>
#include <stdio.h>


int main() {
    LOG(TRACE) << "Hello, World!" << " jeff is not here";
    LOG(DEBUG) << "Hello, World!" << " jeff is not here";
    LOG(INFO) << "Hello, World!" << " jeff is here";
    LOG(WARN) << "Hello, World!" << " jeff is here";
    LOG(ERROR) << "Hello, World!" << " jeff is here";
    LOG(FATAL) << "Hello, World!" << " jeff is here";
    clog::default_logger()->set_level(clog::level::trace);
    LOG(DEBUG) << "Hello, World!" << " jeff is here";
    LOG(TRACE) << "Hello, World!" << " jeff is here";
    clog::default_logger()->set_level(clog::level::info);

    LOG_IF(INFO, true) << "Hello, World!" << " jeff is here";
    LOG_IF(INFO, false) << "Hello, World!" << " jeff is not here";
    PLOG(INFO) << "Hello, World!" << " jeff is here";
    auto r = open("non_existent_file", 0);
    PLOG_IF(ERROR, r<=0) << "Failed to open file";
    const char *s = nullptr;
    std::string exists = "exists";
    std::string non_existent = "non_existent";
    size_t eve = 10;
    for (int i = 0; i < 100; ++i) {
        if(i % eve == 0) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        LOG_EVERY_N(INFO, 10) << s;
    }
    exists = "condition exists";
    non_existent = "condition non_existent";
    for (int i = 0; i < 100; ++i) {
        if(i % eve == 0) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        LOG_IF_EVERY_N(INFO, 10, true) << s;
    }
    exists = "pow2 condition exists";
    non_existent = "pow2 condition non_existent";
    for (int i = 0; i < 100; ++i) {
        auto cond = i % 2 == 0;
        if(i % eve == 0) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        LOG_IF_EVERY_N(INFO, 10, cond) << s;
    }
    LOG(INFO) << "LogFirstNState";

    std::string first_exists = "first exists";
    std::string first_non_existent = "first non_existent";
    auto n_limit = 10;
    for (int i = 0; i < 100; ++i) {
        if(i < n_limit) {
            s = first_exists.c_str();
        } else {
            s = first_non_existent.c_str();
        }
        LOG_FIRST_N(INFO, 10) << s;
    }

    first_exists = "duration exists";
    first_non_existent = "duration non_existent";
    for (int i = 0; i < 20; ++i) {
        if(i < 1) {
            s = first_exists.c_str();
        } else {
            s = first_non_existent.c_str();
        }
        LOG_EVERY_T(INFO, 0.1) << s;
    }
    /// LOG ONCE
    exists = "once exists";
    non_existent = "once non_existent";
    for (int i = 0; i < 10; ++i) {
        if(i == 0) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        LOG_ONCE(INFO) << s;

    }
    exists = "condition once exists";
    non_existent = "condition once non_existent";
    size_t cnt = 0;
    for (int i = 0; i < 10; ++i) {
        bool cond = rand() % 2 == 0;
        if(cnt == 0 && cond) {
            s = exists.c_str();
            cnt++;
        } else {
            s = non_existent.c_str();
        }
        LOG_IF_ONCE(INFO, cond) << s;
    }
    //////////////
    /// DLOG *
    //////////////
    LOG(INFO) << "start dlog";
#if !CLOG_DCHECK_IS_ON()
    LOG(INFO) << "CLOG_DCHECK_IS_ON() false";
    DLOG(INFO) << "Hello, World!" << " jeff is should not here";
    DCHECK(false) << "DCHECK(false) jeff is should not here";
#else
    LOG(INFO) << "CLOG_DCHECK_IS_ON() true";
    DLOG(INFO) << "Hello, World!" << " jeff is should here";
    DCHECK(false) << "DCHECK(false) jeff is should here";
    DCHECK_EQ(1, 2) << "DCHECK_EQ(1, 2)";
    DCHECK_EQ(1, 2u);
    class A {};
    A *a = nullptr;
    DCHECK_EQ(nullptr, collie::ptr(a));
#endif
    DLOG_IF(INFO, true) << "DLOG_IF(INFO, true)" << " jeff is should here";
    LOG(INFO) << "Hello, World!" << " going to exit";
    /// CHECK
    CHECK_EQ(1, 1);
    CHECK_EQ(1, 2) << "CHECK_EQ(1, 2)";
    CHECK_DOUBLE_EQ(1.0, 1.0);
    CHECK_NEAR(1.0, 1.1, 0.2);
    // CHECK_NEAR false
    CHECK_NEAR(1.0, 1.1, 0.05);
    /// vlog
    clog::default_logger()->set_vlog_level(1);
    VLOG(1) << "VLOG(1)"<< " jeff should not here";
    VLOG(0) << "VLOG(0)"<< " jeff should here";
    clog::default_logger()->set_vlog_level(2);
    VLOG(-1) << "VLOG(-1)"<< " jeff should here";
    VLOG(0) << "VLOG(0)"<< " jeff should here";
    VLOG(1) << "VLOG(1)"<< " jeff should here";
    VLOG(2) << "VLOG(2)"<< " jeff should not here";
    VLOG(3) << "VLOG(3)"<< " jeff should not here";
    VLOG_IF(1, true) << "VLOG_IF(1, true)"<< " jeff should here";
    VLOG_IF(1, false) << "VLOG_IF(1, false)"<< " jeff should not here";
    VLOG_IF(2, true) << "VLOG_IF(2, true)"<< " jeff should not here";

    exists = "VLOG_EVERY_N exists";
    non_existent = "VLOG_EVERY_N non_existent";
    for (int i = 0; i < 100; ++i) {
        if(i % eve == 0) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        VLOG_EVERY_N(1, 10) << s;
    }

    exists = "VLOG_IF_EVERY_N exists";
    non_existent = "VLOG_IF_EVERY_N non_existent";
    for (int i = 0; i < 100; ++i) {
        if(i % eve == 0) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        VLOG_IF_EVERY_N(1, 10, true) << s;
    }
    exists = "VLOG_IF_EVERY_N exists";
    non_existent = "VLOG_IF_EVERY_N non_existent";
    for (int i = 0; i < 100; ++i) {
        auto cond = i % 2 == 0;
        if(i % eve == 0) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        VLOG_IF_EVERY_N(1, 10, cond) << s;
    }
    exists = "VLOG_FIRST_N exists";
    non_existent = "VLOG_FIRST_N non_existent";
    for (int i = 0; i < 100; ++i) {
        if(i < n_limit) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        VLOG_FIRST_N(1, 10) << s;
    }

    exists = "VLOG_EVERY_T exists";
    non_existent = "VLOG_EVERY_T non_existent";
    for (int i = 0; i < 20; ++i) {
        if(i < 1) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        VLOG_EVERY_T(1, 0.1) << s;
    }

    exists = "VLOG_ONCE exists";
    non_existent = "VLOG_ONCE non_existent";
    for (int i = 0; i < 10; ++i) {
        if(i == 0) {
            s = exists.c_str();
        } else {
            s = non_existent.c_str();
        }
        VLOG_ONCE(1) << s;
    }
    clog::default_logger()->set_pattern("%v");
    LOG(INFO) << "Hello, World!" << " jeff is here";
    LOG(WARN) << "Hello, World!" << " jeff is here";
    return 0;
}