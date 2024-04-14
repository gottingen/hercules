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

#include "turbo/times/clock.h"
#include "turbo/format/print.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

#define TURBO_MEMORY_RESOURCE_POOL_TEST
#include "turbo/memory/resource_pool.h"
#include "turbo/times/stop_watcher.h"

namespace {
    struct MyObject {};

    int nfoo_dtor = 0;
    struct Foo {
        Foo() {
            x = rand() % 2;
        }
        ~Foo() {
            ++nfoo_dtor;
        }
        int x;
    };
}

namespace turbo {
    template<>
    struct ResourcePoolTraits<MyObject> : public ResourcePoolTraitsBase<MyObject>{
        static constexpr size_t block_max_size() {
            return 128;
        }

        static constexpr size_t block_max_items() {
            return 3;
        }

        static constexpr size_t free_chunk_max_items() {
            return 5;
        }

    };
    template<>
    struct ResourcePoolTraits<Foo> : public ResourcePoolTraitsBase<Foo> {
        constexpr  static bool validate(const Foo *ptr) {
            return ptr->x != 0;
        }
    };
}

namespace {
    using namespace turbo;

    class ResourcePoolTest {
    protected:
        ResourcePoolTest(){
        };
        virtual ~ResourcePoolTest(){};
        virtual void SetUp() {
            srand(time(0));
        };
        virtual void TearDown() {
        };
    };

    TEST_CASE_FIXTURE(ResourcePoolTest, "atomic_array_init") {
        for (int i = 0; i < 2; ++i) {
            if (i == 0) {
                std::atomic<int> a[2];
                a[0] = 1;
                // The folowing will cause compile error with gcc3.4.5 and the
                // reason is unknown
                // a[1] = 2;
            } else if (i == 2) {
                std::atomic<int> a[2];
                REQUIRE_EQ(0, a[0]);
                REQUIRE_EQ(0, a[1]);
            }
        }
    }

    int nc = 0;
    int nd = 0;
    std::set<void*> ptr_set;
    struct YellObj {
        YellObj() {
            ++nc;
            ptr_set.insert(this);
            ::printf("Created %p\n", this);
        }
        ~YellObj() {
            ++nd;
            ptr_set.erase(this);
            ::printf("Destroyed %p\n", this);
        }
        char _dummy[96];
    };

    TEST_CASE_FIXTURE(ResourcePoolTest, "change_config") {
        int a[2];
        printf("%lu\n", TURBO_ARRAY_SIZE(a));

        ResourcePoolInfo info = describe_resources<MyObject>();
        ResourcePoolInfo zero_info = { 0, 0, 0, 0, 3, 3, 0 };
        REQUIRE_EQ(0, memcmp(&info, &zero_info, sizeof(info)));

        ResourceId<MyObject> id = { 0 };
        get_resource<MyObject>(&id);
        std::cout << describe_resources<MyObject>() << std::endl;
        return_resource(id);
        std::cout << describe_resources<MyObject>() << std::endl;
    }

    struct NonDefaultCtorObject {
        explicit NonDefaultCtorObject(int value) : _value(value) {}
        NonDefaultCtorObject(int value, int dummy) : _value(value + dummy) {}

        int _value;
    };

    TEST_CASE_FIXTURE(ResourcePoolTest, "sanity") {
        ptr_set.clear();
        ResourceId<NonDefaultCtorObject> id0 = { 0 };
        get_resource<NonDefaultCtorObject>(&id0, 10);
        REQUIRE_EQ(10, address_resource(id0)->_value);
        get_resource<NonDefaultCtorObject>(&id0, 100, 30);
        REQUIRE_EQ(130, address_resource(id0)->_value);

        ::printf("BLOCK_NITEM=%lu\n", ResourcePool<YellObj>::BLOCK_NITEM);

        nc = 0;
        nd = 0;
        {
            ResourceId<YellObj> id1;
            YellObj* o1 = get_resource(&id1);
            REQUIRE(o1);
            REQUIRE_EQ(o1, address_resource(id1));

            REQUIRE_EQ(1, nc);
            REQUIRE_EQ(0, nd);

            ResourceId<YellObj> id2;
            YellObj* o2 = get_resource(&id2);
            REQUIRE(o2);
            REQUIRE_EQ(o2, address_resource(id2));

            REQUIRE_EQ(2, nc);
            REQUIRE_EQ(0, nd);

            return_resource(id1);
            REQUIRE_EQ(2, nc);
            REQUIRE_EQ(0, nd);

            return_resource(id2);
            REQUIRE_EQ(2, nc);
            REQUIRE_EQ(0, nd);
        }
        REQUIRE_EQ(0, nd);

        clear_resources<YellObj>();
        REQUIRE_EQ(2, nd);
        REQUIRE(ptr_set.empty());
    }

    TEST_CASE_FIXTURE(ResourcePoolTest, "validator") {
        nfoo_dtor = 0;
        int nfoo = 0;
        for (int i = 0; i < 100; ++i) {
            ResourceId<Foo> id = { 0 };
            Foo* foo = get_resource<Foo>(&id);
            if (foo) {
                REQUIRE_EQ(1, foo->x);
                ++nfoo;
            }
        }
        turbo::println("nfoo={}, nfoo_dtor={}", nfoo, nfoo_dtor);
        REQUIRE_EQ(nfoo + nfoo_dtor, 100);
        REQUIRE_EQ((size_t)nfoo, describe_resources<Foo>().item_num);
    }

    TEST_CASE_FIXTURE(ResourcePoolTest, "get_int") {
        clear_resources<int>();

        // Perf of this test is affected by previous case.
        const size_t N = 100000;

        turbo::StopWatcher tm;
        ResourceId<int> id;

        // warm up
        if (get_resource(&id)) {
            return_resource(id);
        }
        REQUIRE_EQ(0UL, id);
        delete (new int);

        tm.reset();
        for (size_t i = 0; i < N; ++i) {
            *get_resource(&id) = i;
        }
        tm.stop();
        ::printf("get a int takes %.1fns\n", tm.elapsed_nano()/(double)N);

        tm.reset();
        for (size_t i = 0; i < N; ++i) {
            *(new int) = i;
        }
        tm.stop();
        ::printf("new a int takes %luns\n", tm.elapsed_nano()/N);

        tm.reset();
        for (size_t i = 0; i < N; ++i) {
            id.value = i;
            *ResourcePool<int>::unsafe_address_resource(id) = i;
        }
        tm.stop();
        ::printf("unsafe_address a int takes %.1fns\n", tm.elapsed_nano()/(double)N);

        tm.reset();
        for (size_t i = 0; i < N; ++i) {
            id.value = i;
            *address_resource(id) = i;
        }
        tm.stop();
        ::printf("address a int takes %.1fns\n", tm.elapsed_nano()/(double)N);

        std::cout << describe_resources<int>() << std::endl;
        clear_resources<int>();
        std::cout << describe_resources<int>() << std::endl;
    }


    struct SilentObj {
        char buf[sizeof(YellObj)];
    };

    TEST_CASE_FIXTURE(ResourcePoolTest, "get_perf") {
        const size_t N = 10000;
        std::vector<SilentObj*> new_list;
        new_list.reserve(N);
        ResourceId<SilentObj> id;

        turbo::StopWatcher tm1, tm2;

        // warm up
        if (get_resource(&id)) {
            return_resource(id);
        }
        delete (new SilentObj);

        // Run twice, the second time will be must faster.
        for (size_t j = 0; j < 2; ++j) {

            tm1.reset();
            for (size_t i = 0; i < N; ++i) {
                get_resource(&id);
            }
            tm1.stop();
            ::printf("get a SilentObj takes %luns\n", tm1.elapsed_nano()/N);
            //clear_resources<SilentObj>(); // free all blocks

            tm2.reset();
            for (size_t i = 0; i < N; ++i) {
                new_list.push_back(new SilentObj);
            }
            tm2.stop();
            ::printf("new a SilentObj takes %luns\n", tm2.elapsed_nano()/N);
            for (size_t i = 0; i < new_list.size(); ++i) {
                delete new_list[i];
            }
            new_list.clear();
        }

        std::cout << describe_resources<SilentObj>() << std::endl;
    }

    struct D { int val[1]; };

    void* get_and_return_int(void*) {
        // Perf of this test is affected by previous case.
        const size_t N = 100000;
        std::vector<ResourceId<D> > v;
        v.reserve(N);
        turbo::StopWatcher tm0, tm1, tm2;
        ResourceId<D> id = {0};
        D tmp = D();
        int sr = 0;

        // warm up
        tm0.reset();
        if (get_resource(&id)) {
            return_resource(id);
        }
        tm0.stop();

        turbo::Println("[{}] warmup={}", pthread_self(), tm0.elapsed_nano());

        for (int j = 0; j < 5; ++j) {
            v.clear();
            sr = 0;

            tm1.reset();
            for (size_t i = 0; i < N; ++i) {
                *get_resource(&id) = tmp;
                v.push_back(id);
            }
            tm1.stop();

            std::random_shuffle(v.begin(), v.end());

            tm2.reset();
            for (size_t i = 0; i < v.size(); ++i) {
                sr += return_resource(v[i]);
            }
            tm2.stop();

            if (0 != sr) {
                turbo::Println("{} return_resource failed", sr);
            }

            turbo::Println("[{}:{}] get<D>={} return<D>={}",
                   pthread_self(), j, tm1.elapsed_nano()/(double)N, tm2.elapsed_nano()/(double)N);
        }
        return nullptr;
    }

    void* new_and_delete_int(void*) {
        const size_t N = 100000;
        std::vector<D*> v2;
        v2.reserve(N);
        turbo::StopWatcher tm0, tm1, tm2;
        D tmp = D();

        for (int j = 0; j < 3; ++j) {
            v2.clear();

            // warm up
            delete (new D);

            tm1.reset();
            for (size_t i = 0; i < N; ++i) {
                D *p = new D;
                *p = tmp;
                v2.push_back(p);
            }
            tm1.stop();

            std::random_shuffle(v2.begin(), v2.end());

            tm2.reset();
            for (size_t i = 0; i < v2.size(); ++i) {
                delete v2[i];
            }
            tm2.stop();

            turbo::Println("[{}:{}] new<D>={} delete<D>={}",
                   pthread_self(), j, tm1.elapsed_nano()/(double)N, tm2.elapsed_nano()/(double)N);
        }

        return nullptr;
    }

    TEST_CASE_FIXTURE(ResourcePoolTest, "get_and_return_int_single_thread") {
        get_and_return_int(nullptr);
        new_and_delete_int(nullptr);
    }

    TEST_CASE_FIXTURE(ResourcePoolTest, "get_and_return_int_multiple_threads") {
        pthread_t tid[16];
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(tid); ++i) {
            REQUIRE_EQ(0, pthread_create(&tid[i], nullptr, get_and_return_int, nullptr));
        }
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(tid); ++i) {
            pthread_join(tid[i], nullptr);
        }

        pthread_t tid2[16];
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(tid2); ++i) {
            REQUIRE_EQ(0, pthread_create(&tid2[i], nullptr, new_and_delete_int, nullptr));
        }
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(tid2); ++i) {
            pthread_join(tid2[i], nullptr);
        }

        std::cout << describe_resources<D>() << std::endl;
        clear_resources<D>();
        ResourcePoolInfo info = describe_resources<D>();
        ResourcePoolInfo zero_info = { 0, 0, 0, 0,
                                       ResourcePoolTraits<D>::block_max_items(),
                                       ResourcePoolTraits<D>::block_max_items(), 0};
        REQUIRE_EQ(0, memcmp(&info, &zero_info, sizeof(info) - sizeof(size_t)));
    }

    TEST_CASE_FIXTURE(ResourcePoolTest, "verify_get") {
        clear_resources<int>();
        std::cout << describe_resources<int>() << std::endl;

        std::vector<std::pair<int*, ResourceId<int> > > v;
        v.reserve(100000);
        ResourceId<int> id = { 0 };
        for (int i = 0; (size_t)i < v.capacity(); ++i)  {
            int* p = get_resource(&id);
            *p = i;
            v.push_back(std::make_pair(p, id));
        }
        int i;
        for (i = 0; (size_t)i < v.size() && *v[i].first == i; ++i);
        REQUIRE_EQ(v.size(), (size_t)i);
        for (i = 0; (size_t)i < v.size() && v[i].second == (size_t)i; ++i);
        REQUIRE_EQ(v.size(), (size_t)i);// << "i=" << i << ", " << v[i].second;

        clear_resources<int>();
    }
} // namespace
