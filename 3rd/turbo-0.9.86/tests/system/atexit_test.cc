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
//
// Created by jeff on 24-1-9.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

#include <errno.h>
#include "turbo/system/atexit.h"
#include "turbo/memory/object_pool.h"

namespace detail {

    template <typename T> void delete_object(void* arg) {
        delete static_cast<T*>(arg);
    }

    template <typename T>
    class ThreadLocalHelper {
    public:
        inline static T* get() {
            if (__builtin_expect(value != NULL, 1)) {
                return value;
            }
            value = new (std::nothrow) T;
            if (value != NULL) {
                turbo::thread_atexit(delete_object<T>, value);
            }
            return value;
        }
        static TURBO_THREAD_LOCAL T* value;
    };

    template <typename T> TURBO_THREAD_LOCAL T* ThreadLocalHelper<T>::value = nullptr;

}  // namespace detail

template <typename T> inline T* get_thread_local() {
    return detail::ThreadLocalHelper<T>::get();
}

namespace {

    TURBO_THREAD_LOCAL int * dummy = nullptr;
    const size_t NTHREAD = 8;
    static bool processed[NTHREAD+1];
    static bool deleted[NTHREAD+1];
    static bool register_check = false;

    struct YellObj {
        static int nc;
        static int nd;
        YellObj() {
            ++nc;
        }
        ~YellObj() {
            ++nd;
        }
    };
    int YellObj::nc = 0;
    int YellObj::nd = 0;

    static void check_global_variable() {
        turbo::println("check_global_variable");/*
        REQUIRE(processed[NTHREAD]);
        REQUIRE(deleted[NTHREAD]);
        turbo::println("check_global_variable {}", processed[NTHREAD]);
        REQUIRE_EQ(2, YellObj::nc);
        REQUIRE_EQ(2, YellObj::nd);*/
        turbo::println("check_global_variable end");
    }

    class AtexitThreadLocalTest{
    protected:
        AtexitThreadLocalTest(){
            if (!register_check) {
                register_check = true;
                turbo::thread_atexit(check_global_variable);
            }
        };
        virtual ~AtexitThreadLocalTest(){};
    };


    TURBO_THREAD_LOCAL void* x;

    void* foo(void* arg) {
        x = arg;
        usleep(10000);
        printf("x=%p\n", x);
        return nullptr;
    }

    TEST_CASE_FIXTURE(AtexitThreadLocalTest, "thread_local_keyword") {
        pthread_t th[2];
        pthread_create(&th[0], nullptr, foo, (void*)1);
        pthread_create(&th[1], nullptr, foo, (void*)2);
        pthread_join(th[0], nullptr);
        pthread_join(th[1], nullptr);
    }

    void* yell(void*) {
        YellObj* p = get_thread_local<YellObj>();
        REQUIRE(p);
        REQUIRE_EQ(2, YellObj::nc);
        REQUIRE_EQ(0, YellObj::nd);
        REQUIRE_EQ(p, get_thread_local<YellObj>());
        REQUIRE_EQ(2, YellObj::nc);
        REQUIRE_EQ(0, YellObj::nd);
        return nullptr;
    }

    TEST_CASE_FIXTURE(AtexitThreadLocalTest, "get_thread_local") {
        YellObj::nc = 0;
        YellObj::nd = 0;
        YellObj* p = get_thread_local<YellObj>();
        REQUIRE(p);
        REQUIRE_EQ(1, YellObj::nc);
        REQUIRE_EQ(0, YellObj::nd);
        REQUIRE_EQ(p, get_thread_local<YellObj>());
        REQUIRE_EQ(1, YellObj::nc);
        REQUIRE_EQ(0, YellObj::nd);
        pthread_t th;
        REQUIRE_EQ(0, pthread_create(&th, nullptr, yell, nullptr));
        pthread_join(th, nullptr);
        REQUIRE_EQ(2, YellObj::nc);
        REQUIRE_EQ(1, YellObj::nd);
    }

    void delete_dummy(void* arg) {
        *(bool*)arg = true;
        if (dummy) {
            delete dummy;
            dummy = nullptr;
        } else {
            printf("dummy is nullptr\n");
        }
    }

    void* proc_dummy(void* arg) {
        bool *p = (bool*)arg;
        *p = true;
        REQUIRE(dummy == nullptr);
        dummy = new int(p - processed);
        turbo::thread_atexit(delete_dummy, deleted + (p - processed));
        return nullptr;
    }

    TEST_CASE_FIXTURE(AtexitThreadLocalTest, "sanity") {
        errno = 0;
        REQUIRE_EQ(-1, turbo::thread_atexit(nullptr));
        REQUIRE_EQ(EINVAL, errno);

        processed[NTHREAD] = false;
        deleted[NTHREAD] = false;
        proc_dummy(&processed[NTHREAD]);

        pthread_t th[NTHREAD];
        for (size_t i = 0; i < NTHREAD; ++i) {
            processed[i] = false;
            deleted[i] = false;
            REQUIRE_EQ(0, pthread_create(&th[i], nullptr, proc_dummy, processed + i));
        }
        for (size_t i = 0; i < NTHREAD; ++i) {
            REQUIRE_EQ(0, pthread_join(th[i], nullptr));
            REQUIRE(processed[i]);
            REQUIRE(deleted[i]);
        }
    }

    static std::ostringstream* oss = nullptr;
    inline std::ostringstream& get_oss() {
        if (oss == nullptr) {
            oss = new std::ostringstream;
        }
        return *oss;
    }

    void fun1() {
        get_oss() << "fun1" << std::endl;
    }

    void fun2() {
        get_oss() << "fun2" << std::endl;
    }

    void fun3(void* arg) {
        get_oss() << "fun3(" << (uintptr_t)arg << ")" << std::endl;
    }

    void fun4(void* arg) {
        get_oss() << "fun4(" << (uintptr_t)arg << ")" << std::endl;
    }

    static void check_result() {
        // Don't use gtest function since this function might be invoked when the main
        // thread quits, instances required by gtest functions are likely destroyed.
        assert(get_oss().str() == "fun4(0)\nfun3(2)\nfun2\n");
    }

    TEST_CASE_FIXTURE(AtexitThreadLocalTest, "call_order_and_cancel") {
        turbo::thread_atexit_cancel(nullptr);
        turbo::thread_atexit_cancel(nullptr, nullptr);

        REQUIRE_EQ(0, turbo::thread_atexit(check_result));

        REQUIRE_EQ(0, turbo::thread_atexit(fun1));
        REQUIRE_EQ(0, turbo::thread_atexit(fun1));
        REQUIRE_EQ(0, turbo::thread_atexit(fun2));
        REQUIRE_EQ(0, turbo::thread_atexit(fun3, (void*)1));
        REQUIRE_EQ(0, turbo::thread_atexit(fun3, (void*)1));
        REQUIRE_EQ(0, turbo::thread_atexit(fun3, (void*)2));
        REQUIRE_EQ(0, turbo::thread_atexit(fun4, nullptr));

        turbo::thread_atexit_cancel(nullptr);
        turbo::thread_atexit_cancel(nullptr, nullptr);
        turbo::thread_atexit_cancel(fun1);
        turbo::thread_atexit_cancel(fun3, nullptr);
        turbo::thread_atexit_cancel(fun3, (void*)1);
    }

} // namespace
