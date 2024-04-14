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

#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <fcntl.h>
#include <turbo/container/flat_hash_map.h>
#include <turbo/container/flat_hash_set.h>
#include "turbo/times/stop_watcher.h"
#include <turbo/system/io.h>
#include <turbo/files/sys/temp_file.h>
#include "turbo/random/random.h"
#include "turbo/system/atexit.h"
#include "turbo/files/filesystem.h"
#include "turbo/log/logging.h"

namespace turbo::files_internal {
    namespace iobuf {
        extern void* (*blockmem_allocate)(size_t);
        extern void (*blockmem_deallocate)(void*);
        extern void reset_blockmem_allocate_and_deallocate();
        extern int32_t block_shared_count(turbo::IOBuf::Block const* b);
        extern uint32_t block_cap(turbo::IOBuf::Block const* b);
        extern IOBuf::Block* get_tls_block_head();
        extern int get_tls_block_count();
        extern void remove_tls_block_chain();
        extern IOBuf::Block* acquire_tls_block();
        extern IOBuf::Block* share_tls_block();
        extern void release_tls_block_chain(IOBuf::Block* b);
        extern uint32_t block_cap(IOBuf::Block const* b);
        extern uint32_t block_size(IOBuf::Block const* b);
        extern IOBuf::Block* get_portal_next(IOBuf::Block const* b);
    }
}

namespace {

    const size_t BLOCK_OVERHEAD = 32; //impl dependent
    const size_t DEFAULT_PAYLOAD = turbo::IOBuf::DEFAULT_BLOCK_SIZE - BLOCK_OVERHEAD;

    void check_tls_block() {
        //REQUIRE_EQ((turbo::IOBuf::Block*)nullptr, turbo::files_internal::iobuf::get_tls_block_head());
        turbo::println("tls_block of turbo::IOBuf was deleted {}", std::this_thread::get_id());
    }
    [[maybe_unused]] const int check_dummy = turbo::thread_atexit(check_tls_block);

    static turbo::flat_hash_set<void*> s_set;

    void* debug_block_allocate(size_t block_size) {
        void* b = operator new (block_size, std::nothrow);
        s_set.insert(b);
        return b;
    }

    void debug_block_deallocate(void* b) {
        if (1ul != s_set.erase(b)) {
            REQUIRE(false);
        } else {
            operator delete(b);
        }
    }

    inline bool is_debug_allocator_enabled() {
        return (turbo::files_internal::iobuf::blockmem_allocate == debug_block_allocate);
    }

    void install_debug_allocator() {
        if (!is_debug_allocator_enabled()) {
            turbo::files_internal::iobuf::remove_tls_block_chain();
            turbo::files_internal::iobuf::blockmem_allocate = debug_block_allocate;
            turbo::files_internal::iobuf::blockmem_deallocate = debug_block_deallocate;
            TLOG_INFO("<Installed debug create/destroy>");
        }
    }

    void show_prof_and_rm(const char* bin_name, const char* filename, size_t topn) {
        char cmd[1024];
        if (topn != 0) {
            snprintf(cmd, sizeof(cmd), "if [ -e %s ] ; then CPUPROFILE_FREQUENCY=1000 ./pprof --text %s %s | head -%lu; rm -f %s; fi", filename, bin_name, filename, topn+1, filename);
        } else {
            snprintf(cmd, sizeof(cmd), "if [ -e %s ] ; then CPUPROFILE_FREQUENCY=1000 ./pprof --text %s %s; rm -f %s; fi", filename, bin_name, filename, filename);
        }
        REQUIRE_EQ(0, system(cmd));
    }

    static void check_memory_leak() {
        if (is_debug_allocator_enabled()) {
            turbo::IOBuf::Block* p = turbo::files_internal::iobuf::get_tls_block_head();
            size_t n = 0;
            while (p) {
                auto it = s_set.find(p);
                REQUIRE_NE(it , s_set.end());
                p = turbo::files_internal::iobuf::get_portal_next(p);
                ++n;
            }
            REQUIRE_EQ(n, s_set.size());
            REQUIRE_EQ(n, (size_t)turbo::files_internal::iobuf::get_tls_block_count());
        }
    }

    class IOBufTest{
    protected:
        IOBufTest(){};
        virtual ~IOBufTest(){
            check_memory_leak();
        };
    };

    std::string to_str(const turbo::IOBuf& p) {
        return p.to_string();
    }

    TEST_CASE_FIXTURE(IOBufTest, "append_zero") {
        int fds[2];
        REQUIRE_EQ(0, pipe(fds));
        turbo::IOPortal p;
        REQUIRE_EQ(0, p.append_from_file_descriptor(fds[0], 0).value());
        REQUIRE_EQ(0, close(fds[0]));
        REQUIRE_EQ(0, close(fds[1]));
    }

    TEST_CASE_FIXTURE(IOBufTest, "pop_front") {
        install_debug_allocator();

        turbo::IOBuf buf;
        REQUIRE_EQ(0UL, buf.pop_front(1));   // nothing happened

        std::string s = "hello";
        buf.append(s);
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(0UL, buf.pop_front(0));   // nothing happened
        REQUIRE_EQ(s, to_str(buf));

        REQUIRE_EQ(1UL, buf.pop_front(1));
        s.erase(0, 1);
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(s.length(), buf.length());
        REQUIRE_FALSE(buf.empty());

        REQUIRE_EQ(s.length(), buf.pop_front(INT_MAX));
        s.clear();
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(0UL, buf.length());
        REQUIRE(buf.empty());

        for (size_t i = 0; i < DEFAULT_PAYLOAD * 3/2; ++i) {
            s.push_back(i);
        }
        buf.append(s);
        REQUIRE_EQ(1UL, buf.pop_front(1));
        s.erase(0, 1);
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(s.length(), buf.length());
        REQUIRE_FALSE(buf.empty());

        REQUIRE_EQ(s.length(), buf.pop_front(INT_MAX));
        s.clear();
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(0UL, buf.length());
        REQUIRE(buf.empty());
    }

    TEST_CASE_FIXTURE(IOBufTest, "pop_back") {
        install_debug_allocator();

        turbo::IOBuf buf;
        REQUIRE_EQ(0UL, buf.pop_back(1));   // nothing happened

        std::string s = "hello";
        buf.append(s);
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(0UL, buf.pop_back(0));   // nothing happened
        REQUIRE_EQ(s, to_str(buf));

        REQUIRE_EQ(1UL, buf.pop_back(1));
        s.resize(s.size() - 1);
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(s.length(), buf.length());
        REQUIRE_FALSE(buf.empty());

        REQUIRE_EQ(s.length(), buf.pop_back(INT_MAX));
        s.clear();
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(0UL, buf.length());
        REQUIRE(buf.empty());

        for (size_t i = 0; i < DEFAULT_PAYLOAD * 3/2; ++i) {
            s.push_back(i);
        }
        buf.append(s);
        REQUIRE_EQ(1UL, buf.pop_back(1));
        s.resize(s.size() - 1);
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(s.length(), buf.length());
        REQUIRE_FALSE(buf.empty());

        REQUIRE_EQ(s.length(), buf.pop_back(INT_MAX));
        s.clear();
        REQUIRE_EQ(s, to_str(buf));
        REQUIRE_EQ(0UL, buf.length());
        REQUIRE(buf.empty());
    }

    TEST_CASE_FIXTURE(IOBufTest, "append") {
        install_debug_allocator();

        turbo::IOBuf b;
        REQUIRE_EQ(0UL, b.length());
        REQUIRE(b.empty());
        REQUIRE_EQ(-1, b.append(nullptr));
        REQUIRE_EQ(0, b.append(""));
        REQUIRE_EQ(0, b.append(std::string()));
        REQUIRE_EQ(-1, b.append(nullptr, 1));
        REQUIRE_EQ(0, b.append("dummy", 0));
        REQUIRE_EQ(0UL, b.length());
        REQUIRE(b.empty());
        REQUIRE_EQ(0, b.append("1"));
        REQUIRE_EQ(1UL, b.length());
        REQUIRE_FALSE(b.empty());
        REQUIRE_EQ("1", to_str(b));
        const std::string s = "22";
        REQUIRE_EQ(0, b.append(s));
        REQUIRE_EQ(3UL, b.length());
        REQUIRE_FALSE(b.empty());
        REQUIRE_EQ("122", to_str(b));
    }

    TEST_CASE_FIXTURE(IOBufTest, "appendv") {
        install_debug_allocator();

        turbo::IOBuf b;
        const_iovec vec[] = { {"hello1", 6}, {" world1", 7},
                              {"hello2", 6}, {" world2", 7},
                              {"hello3", 6}, {" world3", 7},
                              {"hello4", 6}, {" world4", 7},
                              {"hello5", 6}, {" world5", 7} };
        REQUIRE_EQ(0, b.appendv(vec, TURBO_ARRAY_SIZE(vec)));
        REQUIRE_EQ("hello1 world1hello2 world2hello3 world3hello4 world4hello5 world5",
                  b.to_string());

        // Make some iov_len shorter to test if iov_len works.
        vec[2].iov_len = 4;  // "hello2"
        vec[5].iov_len = 3;  // " world3"
        b.clear();
        REQUIRE_EQ(0, b.appendv(vec, TURBO_ARRAY_SIZE(vec)));
        REQUIRE_EQ("hello1 world1hell world2hello3 wohello4 world4hello5 world5",
                  b.to_string());

        // Append some long stuff.
        const size_t full_len = DEFAULT_PAYLOAD * 9;
        char* str = (char*)malloc(full_len);
        REQUIRE(str);
        const size_t len1 = full_len / 6;
        const size_t len2 = full_len / 3;
        const size_t len3 = full_len - len1 - len2;
        REQUIRE_GT(len1, (size_t)DEFAULT_PAYLOAD);
        REQUIRE_GT(len2, (size_t)DEFAULT_PAYLOAD);
        REQUIRE_GT(len3, (size_t)DEFAULT_PAYLOAD);
        REQUIRE_EQ(full_len, len1 + len2 + len3);

        for (size_t i = 0; i < full_len; ++i) {
            str[i] = i * 7;
        }
        const_iovec vec2[] = {{str, len1},
                              {str + len1, len2},
                              {str + len1 + len2, len3}};
        b.clear();
        REQUIRE_EQ(0, b.appendv(vec2, TURBO_ARRAY_SIZE(vec2)));
        REQUIRE_EQ(full_len, b.size());
        REQUIRE_EQ(0, memcmp(str, b.to_string().data(), full_len));
    }

    TEST_CASE_FIXTURE(IOBufTest, "reserve") {
        turbo::IOBuf b;
        REQUIRE_EQ(turbo::IOBuf::INVALID_AREA, b.reserve(0));
        const size_t NRESERVED1 = 5;
        const turbo::IOBuf::Area a1 = b.reserve(NRESERVED1);
        REQUIRE(a1 != turbo::IOBuf::INVALID_AREA);
        REQUIRE_EQ(NRESERVED1, b.size());
        b.append("hello world");
        REQUIRE_EQ(0, b.unsafe_assign(a1, "prefix")); // `x' will not be copied
        REQUIRE_EQ("prefihello world", b.to_string());
        REQUIRE_EQ((size_t)16, b.size());

        // pop/append sth. from back-side and assign again.
        REQUIRE_EQ((size_t)5, b.pop_back(5));
        REQUIRE_EQ("prefihello ", b.to_string());
        b.append("blahblahfoobar");
        REQUIRE_EQ(0, b.unsafe_assign(a1, "goodorbad")); // `x' will not be copied
        REQUIRE_EQ("goodohello blahblahfoobar", b.to_string());

        // append a long string and assign again.
        std::string s1(DEFAULT_PAYLOAD * 3, '\0');
        for (size_t i = 0; i < s1.size(); ++i) {
            s1[i] = i * 7;
        }
        REQUIRE_EQ(DEFAULT_PAYLOAD * 3, s1.size());
        // remove everything after reserved area
        REQUIRE_GE(b.size(), NRESERVED1);
        b.pop_back(b.size() - NRESERVED1);
        REQUIRE_EQ(NRESERVED1, b.size());
        b.append(s1);
        REQUIRE_EQ(0, b.unsafe_assign(a1, "appleblahblah"));
        REQUIRE_EQ("apple" + s1, b.to_string());

        // Reserve long
        b.pop_back(b.size() - NRESERVED1);
        REQUIRE_EQ(NRESERVED1, b.size());
        const size_t NRESERVED2 = DEFAULT_PAYLOAD * 3;
        const turbo::IOBuf::Area a2 = b.reserve(NRESERVED2);
        REQUIRE_EQ(NRESERVED1 + NRESERVED2, b.size());
        b.append(s1);
        REQUIRE_EQ(NRESERVED1 + NRESERVED2 + s1.size(), b.size());
        std::string s2(NRESERVED2, 0);
        for (size_t i = 0; i < s2.size(); ++i) {
            s2[i] = i * 13;
        }
        REQUIRE_EQ(NRESERVED2, s2.size());
        REQUIRE_EQ(0, b.unsafe_assign(a2, s2.data()));
        REQUIRE_EQ("apple" + s2 + s1, b.to_string());
        REQUIRE_EQ(0, b.unsafe_assign(a1, "orangeblahblah"));
        REQUIRE_EQ("orang" + s2 + s1, b.to_string());
    }

    struct FakeBlock {
        int nshared;
        FakeBlock() : nshared(1) {}
    };

    TEST_CASE_FIXTURE(IOBufTest, "iobuf_as_queue") {
        install_debug_allocator();

        // If INITIAL_CAP gets bigger, creating turbo::IOBuf::Block are very
        // small. Since We don't access turbo::IOBuf::Block::data in this case.
        // We replace turbo::IOBuf::Block with FakeBlock with only nshared (in
        // the same offset)
        FakeBlock* blocks[turbo::IOBuf::INITIAL_CAP+16];
        const size_t NBLOCKS = TURBO_ARRAY_SIZE(blocks);
        turbo::IOBuf::BlockRef r[NBLOCKS];
        const size_t LENGTH = 7UL;
        for (size_t i = 0; i < NBLOCKS; ++i) {
            REQUIRE((blocks[i] = new FakeBlock));
            r[i].offset = 1;
            r[i].length = LENGTH;
            r[i].block = (turbo::IOBuf::Block*)blocks[i];
        }

        turbo::IOBuf p;

        // Empty
        REQUIRE_EQ(0UL, p._ref_num());
        REQUIRE_EQ(-1, p._pop_front_ref());
        REQUIRE_EQ(0UL, p.length());

        // Add one ref
        p._push_back_ref(r[0]);
        REQUIRE_EQ(1UL, p._ref_num());
        REQUIRE_EQ(LENGTH, p.length());
        REQUIRE_EQ(r[0], p._front_ref());
        REQUIRE_EQ(r[0], p._back_ref());
        REQUIRE_EQ(r[0], p._ref_at(0));
        REQUIRE_EQ(2, turbo::files_internal::iobuf::block_shared_count(r[0].block));

        // Add second ref
        p._push_back_ref(r[1]);
        REQUIRE_EQ(2UL, p._ref_num());
        REQUIRE_EQ(LENGTH*2, p.length());
        REQUIRE_EQ(r[0], p._front_ref());
        REQUIRE_EQ(r[1], p._back_ref());
        REQUIRE_EQ(r[0], p._ref_at(0));
        REQUIRE_EQ(r[1], p._ref_at(1));
        REQUIRE_EQ(2, turbo::files_internal::iobuf::block_shared_count(r[1].block));

        // Pop a ref
        REQUIRE_EQ(0, p._pop_front_ref());
        REQUIRE_EQ(1UL, p._ref_num());
        REQUIRE_EQ(LENGTH, p.length());

        REQUIRE_EQ(r[1], p._front_ref());
        REQUIRE_EQ(r[1], p._back_ref());
        REQUIRE_EQ(r[1], p._ref_at(0));
        //REQUIRE_EQ(1, turbo::files_internal::iobuf::block_shared_count(r[0].block));

        // Pop second
        REQUIRE_EQ(0, p._pop_front_ref());
        REQUIRE_EQ(0UL, p._ref_num());
        REQUIRE_EQ(0UL, p.length());
        //REQUIRE_EQ(1, r[1].block->nshared);

        // Add INITIAL_CAP+2 refs, r[0] and r[1] are used, don't use again
        for (size_t i = 0; i < turbo::IOBuf::INITIAL_CAP+2; ++i) {
            p._push_back_ref(r[i+2]);
            REQUIRE_EQ(i+1, p._ref_num());
            REQUIRE_EQ(p._ref_num()*LENGTH, p.length());
            REQUIRE_EQ(r[2], p._front_ref());
            REQUIRE_EQ(r[i+2], p._back_ref());
            for (size_t j = 0; j <= i; j+=std::max(1UL, i/20) /*not check all*/) {
                REQUIRE_EQ(r[j+2], p._ref_at(j));
            }
            REQUIRE_EQ(2, turbo::files_internal::iobuf::block_shared_count(r[i+2].block));
        }

        // Pop them all
        const size_t saved_ref_num = p._ref_num();
        while (p._ref_num() >= 2UL) {
            const size_t last_ref_num = p._ref_num();
            REQUIRE_EQ(0, p._pop_front_ref());
            REQUIRE_EQ(last_ref_num, p._ref_num()+1);
            REQUIRE_EQ(p._ref_num()*LENGTH, p.length());
            const size_t f = saved_ref_num - p._ref_num() + 2;
            REQUIRE_EQ(r[f], p._front_ref());
            REQUIRE_EQ(r[saved_ref_num+1], p._back_ref());
            for (size_t j = 0; j < p._ref_num(); j += std::max(1UL, p._ref_num()/20)) {
                REQUIRE_EQ(r[j+f], p._ref_at(j));
            }
            //REQUIRE_EQ(1, r[f-1].block->nshared);
        }

        REQUIRE_EQ(1ul, p._ref_num());
        // Pop last one
        REQUIRE_EQ(0, p._pop_front_ref());
        REQUIRE_EQ(0UL, p._ref_num());
        REQUIRE_EQ(0UL, p.length());
        //REQUIRE_EQ(1, r[saved_ref_num+1].block->nshared);

        // Delete blocks
        for (size_t i = 0; i < NBLOCKS; ++i) {
            delete blocks[i];
        }
    }

    TEST_CASE_FIXTURE(IOBufTest, "iobuf_sanity") {
        install_debug_allocator();
        
        TLOG_INFO("sizeof(turbo::IOBuf)={} sizeof(IOPortal)={}",
                  sizeof(turbo::IOBuf), sizeof(turbo::IOPortal));

        turbo::IOBuf b1;
        std::string s1 = "hello world";
        const char c1 = 'A';
        const std::string s2 = "too simple";
        std::string s1c = s1;
        s1c.erase(0, 1);

        // Append a c-std::string
        REQUIRE_EQ(0, b1.append(s1.c_str()));
        REQUIRE_EQ(s1.length(), b1.length());
        REQUIRE_EQ(s1, to_str(b1));
        REQUIRE_EQ(1UL, b1._ref_num());

        // Append a char
        REQUIRE_EQ(0, b1.push_back(c1));
        REQUIRE_EQ(s1.length() + 1, b1.length());
        REQUIRE_EQ(s1+c1, to_str(b1));
        REQUIRE_EQ(1UL, b1._ref_num());

        // Append a std::string
        REQUIRE_EQ(0, b1.append(s2));
        REQUIRE_EQ(s1.length() + 1 + s2.length(), b1.length());
        REQUIRE_EQ(s1+c1+s2, to_str(b1));
        REQUIRE_EQ(1UL, b1._ref_num());

        // Pop first char
        REQUIRE_EQ(1UL, b1.pop_front(1));
        REQUIRE_EQ(1UL, b1._ref_num());
        REQUIRE_EQ(s1.length() + s2.length(), b1.length());
        REQUIRE_EQ(s1c+c1+s2, to_str(b1));

        // Pop all
        REQUIRE_EQ(0UL, b1.pop_front(0));
        REQUIRE_EQ(s1.length() + s2.length(), b1.pop_front(b1.length()+1));
        REQUIRE(b1.empty());
        REQUIRE_EQ(0UL, b1.length());
        REQUIRE_EQ(0UL, b1._ref_num());
        REQUIRE_EQ("", to_str(b1));

        // Restore
        REQUIRE_EQ(0, b1.append(s1.c_str()));
        REQUIRE_EQ(0, b1.push_back(c1));
        REQUIRE_EQ(0, b1.append(s2));

        // Cut first char
        turbo::IOBuf p;
        b1.cutn(&p, 0);
        b1.cutn(&p, 1);
        REQUIRE_EQ(s1.substr(0, 1), to_str(p));
        REQUIRE_EQ(s1c+c1+s2, to_str(b1));

        // Cut another two and append to p
        b1.cutn(&p, 2);
        REQUIRE_EQ(s1.substr(0, 3), to_str(p));
        std::string s1d = s1;
        s1d.erase(0, 3);
        REQUIRE_EQ(s1d+c1+s2, to_str(b1));

        // Clear
        b1.clear();
        REQUIRE(b1.empty());
        REQUIRE_EQ(0UL, b1.length());
        REQUIRE_EQ(0UL, b1._ref_num());
        REQUIRE_EQ("", to_str(b1));
        REQUIRE_EQ(s1.substr(0, 3), to_str(p));
    }

    TEST_CASE_FIXTURE(IOBufTest, "copy_and_assign") {
        install_debug_allocator();

        const size_t TARGET_SIZE = turbo::IOBuf::DEFAULT_BLOCK_SIZE * 2;
        turbo::IOBuf buf0;
        buf0.append("hello");
        REQUIRE_EQ(1u, buf0._ref_num());

        // Copy-construct from SmallView
        turbo::IOBuf buf1 = buf0;
        REQUIRE_EQ(1u, buf1._ref_num());
        REQUIRE_EQ(buf0, buf1);

        buf1.resize(TARGET_SIZE, 'z');
        REQUIRE_LT(2u, buf1._ref_num());
        REQUIRE_EQ(TARGET_SIZE, buf1.size());

        // Copy-construct from BigView
        turbo::IOBuf buf2 = buf1;
        REQUIRE_EQ(buf1, buf2);

        // assign BigView to SmallView
        turbo::IOBuf buf3;
        buf3 = buf1;
        REQUIRE_EQ(buf1, buf3);

        // assign BigView to BigView
        turbo::IOBuf buf4;
        buf4.resize(TARGET_SIZE, 'w');
        REQUIRE_NE(buf1, buf4);
        buf4 = buf1;
        REQUIRE_EQ(buf1, buf4);
    }

    TEST_CASE_FIXTURE(IOBufTest, "compare") {
        install_debug_allocator();

        const char* SEED = "abcdefghijklmnqopqrstuvwxyz";
        turbo::IOBuf seedbuf;
        seedbuf.append(SEED);
        const int REP = 100;
        turbo::IOBuf b1;
        for (int i = 0; i < REP; ++i) {
            b1.append(seedbuf);
            b1.append(SEED);
        }
        turbo::IOBuf b2;
        for (int i = 0; i < REP * 2; ++i) {
            b2.append(SEED);
        }
        REQUIRE_EQ(b1, b2);

        turbo::IOBuf b3 = b2;

        b2.push_back('0');
        REQUIRE_NE(b1, b2);
        REQUIRE_EQ(b1, b3);

        b1.clear();
        b2.clear();
        REQUIRE_EQ(b1, b2);
    }

    TEST_CASE_FIXTURE(IOBufTest, "append_and_cut_it_all") {
        turbo::IOBuf b;
        const size_t N = 32768UL;
        for (size_t i = 0; i < N; ++i) {
            REQUIRE_EQ(0, b.push_back(i));
        }
        REQUIRE_EQ(N, b.length());
        turbo::IOBuf p;
        b.cutn(&p, N);
        REQUIRE(b.empty());
        REQUIRE_EQ(N, p.length());
    }

    TEST_CASE_FIXTURE(IOBufTest, "copy_to") {
        turbo::IOBuf b;
        const std::string seed = "abcdefghijklmnopqrstuvwxyz";
        std::string src;
        for (size_t i = 0; i < 1000; ++i) {
            src.append(seed);
        }
        b.append(src);
        REQUIRE_GT(b.size(), DEFAULT_PAYLOAD);
        std::string s1;
        REQUIRE_EQ(src.size(), b.copy_to(&s1));
        REQUIRE_EQ(src, s1);

        std::string s2;
        REQUIRE_EQ(32u, b.copy_to(&s2, 32));
        REQUIRE_EQ(src.substr(0, 32), s2);

        std::string s3;
        const std::string expected = src.substr(DEFAULT_PAYLOAD - 1, 33);
        REQUIRE_EQ(33u, b.copy_to(&s3, 33, DEFAULT_PAYLOAD - 1));
        REQUIRE_EQ(expected, s3);

        REQUIRE_EQ(33u, b.append_to(&s3, 33, DEFAULT_PAYLOAD - 1));
        REQUIRE_EQ(expected + expected, s3);

        turbo::IOBuf b1;
        REQUIRE_EQ(src.size(), b.append_to(&b1));
        REQUIRE_EQ(src, b1.to_string());

        turbo::IOBuf b2;
        REQUIRE_EQ(32u, b.append_to(&b2, 32));
        REQUIRE_EQ(src.substr(0, 32), b2.to_string());

        turbo::IOBuf b3;
        REQUIRE_EQ(33u, b.append_to(&b3, 33, DEFAULT_PAYLOAD - 1));
        REQUIRE_EQ(expected, b3.to_string());

        REQUIRE_EQ(33u, b.append_to(&b3, 33, DEFAULT_PAYLOAD - 1));
        REQUIRE_EQ(expected + expected, b3.to_string());
    }

    TEST_CASE_FIXTURE(IOBufTest, "cut_by_single_text_delim") {
        install_debug_allocator();

        turbo::IOBuf b;
        turbo::IOBuf p;
        std::vector<turbo::IOBuf> ps;
        std::string s1 = "1234567\n12\n\n2567";
        REQUIRE_EQ(0, b.append(s1));
        REQUIRE_EQ(s1.length(), b.length());

        for (; b.cut_until(&p, "\n") == 0; ) {
            ps.push_back(p);
            p.clear();
        }

        REQUIRE_EQ(3UL, ps.size());
        REQUIRE_EQ("1234567", to_str(ps[0]));
        REQUIRE_EQ("12", to_str(ps[1]));
        REQUIRE_EQ("", to_str(ps[2]));
        REQUIRE_EQ("2567", to_str(b));

        b.append("\n");
        REQUIRE_EQ(0, b.cut_until(&p, "\n"));
        REQUIRE_EQ("2567", to_str(p));
        REQUIRE_EQ("", to_str(b));
    }

    TEST_CASE_FIXTURE(IOBufTest, "cut_by_multiple_text_delim") {
        install_debug_allocator();

        turbo::IOBuf b;
        turbo::IOBuf p;
        std::vector<turbo::IOBuf> ps;
        std::string s1 = "\r\n1234567\r\n12\r\n\n\r2567";
        REQUIRE_EQ(0, b.append(s1));
        REQUIRE_EQ(s1.length(), b.length());

        for (; b.cut_until(&p, "\r\n") == 0; ) {
            ps.push_back(p);
            p.clear();
        }

        REQUIRE_EQ(3UL, ps.size());
        REQUIRE_EQ("", to_str(ps[0]));
        REQUIRE_EQ("1234567", to_str(ps[1]));
        REQUIRE_EQ("12", to_str(ps[2]));
        REQUIRE_EQ("\n\r2567", to_str(b));

        b.append("\r\n");
        REQUIRE_EQ(0, b.cut_until(&p, "\r\n"));
        REQUIRE_EQ("\n\r2567", to_str(p));
        REQUIRE_EQ("", to_str(b));
    }

    TEST_CASE_FIXTURE(IOBufTest, "append_a_lot_and_cut_them_all") {
        install_debug_allocator();

        turbo::IOBuf b;
        turbo::IOBuf p;
        std::string s1 = "12345678901234567";
        const size_t N = 10000;
        for (size_t i= 0; i < N; ++i) {
            b.append(s1);
        }
        REQUIRE_EQ(N*s1.length(), b.length());

        while (b.length() >= 7) {
            b.cutn(&p, 7);
        }
        size_t remain = s1.length()*N % 7;
        REQUIRE_EQ(remain, b.length());
        REQUIRE_EQ(s1.substr(s1.length() - remain, remain), to_str(b));
        REQUIRE_EQ(s1.length()*N/7*7, p.length());
    }

    TEST_CASE_FIXTURE(IOBufTest, "cut_into_fd_tiny") {
        install_debug_allocator();

        turbo::IOPortal b1, b2;
        std::string ref;
        int fds[2];

        for (int j = 10; j > 0; --j) {
            ref.push_back(j);
        }
        b1.append(ref);
        REQUIRE_EQ(1UL, b1.pop_front(1));
        ref.erase(0, 1);
        REQUIRE_EQ(ref, to_str(b1));
        TLOG_INFO("ref.size={}", ref.size());

        //REQUIRE_EQ(0, pipe(fds));
        REQUIRE_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, 0, fds));

        turbo::make_non_blocking(fds[0]);
        turbo::make_non_blocking(fds[1]);

        while (!b1.empty() || b2.length() != ref.length()) {
            size_t b1len = b1.length(), b2len = b2.length();
            errno = 0;
            turbo::println("b1.length={} - {} ({})", b1len, b1.cut_into_file_descriptor(fds[1]).value(), strerror(errno));
            turbo::println("b2.length={} - {} ({})", b2len, b2.append_from_file_descriptor(fds[0], LONG_MAX).value(), strerror(errno));
        }
        turbo::println("b1.length={}, b2.length={}", b1.length(), b2.length());

        REQUIRE_EQ(ref, to_str(b2));

        close(fds[0]);
        close(fds[1]);
    }

    TEST_CASE_FIXTURE(IOBufTest, "cut_multiple_into_fd_tiny") {
        install_debug_allocator();

        turbo::IOBuf* b1[10];
        turbo::IOPortal b2;
        std::string ref;
        int fds[2];

        for (size_t j = 0; j < TURBO_ARRAY_SIZE(b1); ++j) {
            std::string s;
            for (int i = 10; i > 0; --i) {
                s.push_back(j * 10 + i);
            }
            ref.append(s);
            turbo::IOPortal* b = new turbo::IOPortal();
            b->append(s);
            b1[j] = b;
        }

        REQUIRE_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
        turbo::make_non_blocking(fds[0]);
        turbo::make_non_blocking(fds[1]);

        REQUIRE_EQ((ssize_t)ref.length(),
                  turbo::IOBuf::cut_multiple_into_file_descriptor(
                          fds[1], b1, TURBO_ARRAY_SIZE(b1)).value());
        for (size_t j = 0; j < TURBO_ARRAY_SIZE(b1); ++j) {
            REQUIRE(b1[j]->empty());
            delete (turbo::IOPortal*)b1[j];
            b1[j] = nullptr;
        }
        REQUIRE_EQ((ssize_t)ref.length(),
                  b2.append_from_file_descriptor(fds[0], LONG_MAX).value());
        REQUIRE_EQ(ref, to_str(b2));

        close(fds[0]);
        close(fds[1]);
    }

    TEST_CASE_FIXTURE(IOBufTest, "cut_into_fd_a_lot_of_data") {
        install_debug_allocator();

        turbo::IOPortal b0, b1, b2;
        std::string s, ref;
        int fds[2];

        for (int j = rand()%7777+300000; j > 0; --j) {
            ref.push_back(j);
            s.push_back(j);

            if (s.length() == 1024UL || j == 1) {
                b0.append(s);
                ref.append("tailing");
                b0.cutn(&b1, b0.length());
                REQUIRE_EQ(0, b1.append("tailing"));
                s.clear();
            }
        }

        REQUIRE_EQ(ref.length(), b1.length());
        REQUIRE_EQ(ref, to_str(b1));
        REQUIRE(b0.empty());
        TLOG_INFO("ref.size={}", ref.size());

        //REQUIRE_EQ(0, pipe(fds));
        REQUIRE_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
        turbo::make_non_blocking(fds[0]);
        turbo::make_non_blocking(fds[1]);
        const int sockbufsize = 16 * 1024 - 17;
        REQUIRE_EQ(0, setsockopt(fds[1], SOL_SOCKET, SO_SNDBUF,
                                (const char*)&sockbufsize, sizeof(int)));
        REQUIRE_EQ(0, setsockopt(fds[0], SOL_SOCKET, SO_RCVBUF,
                                (const char*)&sockbufsize, sizeof(int)));

        while (!b1.empty() || b2.length() != ref.length()) {
            size_t b1len = b1.length(), b2len = b2.length();
            errno = 0;
            turbo::println("b1.length={} - {} ({})", b1len, b1.cut_into_file_descriptor(fds[1]).value(), strerror(errno));
            turbo::println("b2.length={} + {} ({})", b2len, b2.append_from_file_descriptor(fds[0], LONG_MAX).value(), strerror(errno));
        }
        turbo::println("b1.length={}, b2.length={}", b1.length(), b2.length());

        REQUIRE_EQ(ref, to_str(b2));

        close(fds[0]);
        close(fds[1]);
    }

    TEST_CASE_FIXTURE(IOBufTest, "cut_by_delim_perf") {
        turbo::files_internal::iobuf::reset_blockmem_allocate_and_deallocate();

        turbo::IOBuf b;
        turbo::IOBuf p;
        std::vector<turbo::IOBuf> ps;
        std::string s1 = "123456789012345678901234567890\n";
        const size_t N = 100000;
        for (size_t i = 0; i < N; ++i) {
            REQUIRE_EQ(0, b.append(s1));
        }
        REQUIRE_EQ(N * s1.length(), b.length());

        turbo::StopWatcher t;
        //ProfilerStart("cutd.prof");
        t.reset();
        for (; b.cut_until(&p, "\n") == 0; ) { }
        t.stop();
        //ProfilerStop();
        TLOG_INFO("IOPortal::cut_by_delim takes {}ns, tp={}MB/s",
                  t.elapsed_nano()/N, s1.length() * N * 1000.0 / t.elapsed_nano ());
        show_prof_and_rm("test_iobuf", "cutd.prof", 10);
    }


    TEST_CASE_FIXTURE(IOBufTest, "cut_perf") {
        turbo::files_internal::iobuf::reset_blockmem_allocate_and_deallocate();

        turbo::IOBuf b;
        turbo::IOBuf p;
        const size_t length = 60000000UL;
        const size_t REP = 10;
        turbo::StopWatcher t;
        std::string s = "1234567890";
        const bool push_char = false;

        if (!push_char) {
            //ProfilerStart("iobuf_append.prof");
            t.reset();
            for (size_t j = 0; j < REP; ++j) {
                b.clear();
                for (size_t i = 0; i < length/s.length(); ++i) {
                    b.append(s);
                }
            }
            t.stop();
            //ProfilerStop();
            TLOG_INFO("IOPortal::append(std::string) takes {}ns, tp={}MB/s",
                      t.elapsed_nano() / length / REP, REP * length * 1000.0 / t.elapsed_nano ());
        } else {
            //ProfilerStart("iobuf_pushback.prof");
            t.reset();
            for (size_t i = 0; i < length; ++i) {
                b.push_back(i);
            }
            t.stop();
            //ProfilerStop();
            TLOG_INFO("IOPortal::push_back(char) takes {}ns, tp={}MB/s",
                      t.elapsed_nano() / length, length * 1000.0 / t.elapsed_nano ());
        }

        REQUIRE_EQ(length, b.length());

        size_t w[3] = { 16, 128, 1024 };
        //char name[32];

        for (size_t i = 0; i < TURBO_ARRAY_SIZE(w); ++i) {
            // snprintf(name, sizeof(name), "iobuf_cut%lu.prof", w[i]);
            // ProfilerStart(name);

            t.reset ();
            while (b.length() >= w[i]+12) {
                b.cutn(&p, 12);
                b.cutn(&p, w[i]);
            }
            t.stop ();

            //ProfilerStop();

            TLOG_INFO("IOPortal::cutn(12+{}) takes {}ns, tp={}MB/s",
                      w[i], t.elapsed_nano()*(w[i]+12)/length, length * 1000.0 / t.elapsed_nano ());

            REQUIRE_LT(b.length(), w[i]+12);

            t.reset();
            b.append(p);
            t.stop();
            TLOG_INFO("IOPortal::append(turbo::IOBuf) takes {}ns, tp={}MB/s",
                      t.elapsed_nano()/p._ref_num(), length * 1000.0 / t.elapsed_nano ());

            p.clear();
            REQUIRE_EQ(length, b.length());
        }

        show_prof_and_rm("test_iobuf", "./iobuf_append.prof", 10);
        show_prof_and_rm("test_iobuf", "./iobuf_pushback.prof", 10);
    }

    TEST_CASE_FIXTURE(IOBufTest, "append_store_append_cut") {
        turbo::files_internal::iobuf::reset_blockmem_allocate_and_deallocate();

        std::string ref;
        ref.resize(rand()%376813+19777777);
        for (size_t j = 0; j < ref.size(); ++j) {
            ref[j] = j;
        }

        turbo::IOPortal b1, b2;
        std::vector<turbo::IOBuf> ps;
        ssize_t nr;
        size_t HINT = 16*1024UL;
        turbo::StopWatcher t;
        size_t w[3] = { 16, 128, 1024 };
        char name[64];
        char profname[64];
        char cmd[512];
        bool write_to_dev_null = true;
        size_t nappend, ncut;

        turbo::TempFile f;
        REQUIRE(f.open().ok());
        REQUIRE_EQ(turbo::ok_status(), f.write(ref.data(), ref.length()));

        for (size_t i = 0; i < TURBO_ARRAY_SIZE(w); ++i) {
            ps.reserve(ref.size()/(w[i]+12) + 1);
            // LOG(INFO) << "ps.cap=" << ps.capacity();

            const int ifd = open(f.path().c_str(), O_RDONLY);
            REQUIRE(ifd > 0);
            if (write_to_dev_null) {
                snprintf(name, sizeof(name), "/dev/null");
            } else {
                ::snprintf(name, sizeof(name), "iobuf_asac%lu.output", w[i]);
                ::snprintf(cmd, sizeof(cmd), "cmp %s %s", f.path().c_str(), name);
            }
            const int ofd = open(name, O_CREAT | O_WRONLY, 0666);
            REQUIRE(ofd > 0);
            snprintf(profname, sizeof(profname), "iobuf_asac%lu.prof", w[i]);

            //ProfilerStart(profname);
            t.reset();

            nappend = ncut = 0;
            while ((nr = b1.append_from_file_descriptor(ifd, HINT).value_or(-1)) > 0) {
                ++nappend;
                while (b1.length() >= w[i] + 12) {
                    turbo::IOBuf p;
                    b1.cutn(&p, 12);
                    b1.cutn(&p, w[i]);
                    ps.push_back(p);
                }
            }
            for (size_t j = 0; j < ps.size(); ++j) {
                b2.append(ps[j]);
                if (b2.length() >= HINT) {
                    REQUIRE(b2.cut_into_file_descriptor(ofd).ok());
                }
            }
            REQUIRE(b2.cut_into_file_descriptor(ofd).ok());
            REQUIRE(b1.cut_into_file_descriptor(ofd).ok());

            close(ifd);
            close(ofd);
            t.stop();
            //ProfilerStop();

            REQUIRE(b1.empty());
            REQUIRE(b2.empty());
            //LOG(INFO) << "ps.size=" << ps.size();
            ps.clear();
            TLOG_INFO("Bandwidth of append({})->cut(12+{})->store->append->cut({}) is {}MB/s",
                      f.path(), w[i], name, ref.length() * 1000.0 / t.elapsed_nano());

            if (!write_to_dev_null) {
                REQUIRE_EQ(0, system(cmd));
            }
            if (!write_to_dev_null) {
                remove(name);
            }
        }

        for (size_t i = 0; i < TURBO_ARRAY_SIZE(w); ++i) {
            snprintf(name, sizeof(name), "iobuf_asac%lu.prof", w[i]);
            show_prof_and_rm("test_iobuf", name, 10);
        }
    }
    
    TEST_CASE_FIXTURE(IOBufTest, "extended_backup") {
        for (int i = 0; i < 2; ++i) {
            std::cout << "i=" << i << std::endl;
            // Consume the left TLS block so that cases are easier to check.
            turbo::files_internal::iobuf::remove_tls_block_chain();
            turbo::IOBuf src;
            const int BLKSIZE = (i == 0 ? 1024 : turbo::IOBuf::DEFAULT_BLOCK_SIZE);
            const int PLDSIZE = BLKSIZE - BLOCK_OVERHEAD;
            turbo::IOBufAsZeroCopyOutputStream out_stream1(&src, BLKSIZE);
            turbo::IOBufAsZeroCopyOutputStream out_stream2(&src);
            turbo::IOBufAsZeroCopyOutputStream & out_stream =
                    (i == 0 ? out_stream1 : out_stream2);
            void* blk1 = nullptr;
            int size1 = 0;
            REQUIRE(out_stream.next(&blk1, &size1));
            REQUIRE_EQ(PLDSIZE, size1);
            REQUIRE_EQ(size1, out_stream.byte_count());
            void* blk2 = nullptr;
            int size2 = 0;
            REQUIRE(out_stream.next(&blk2, &size2));
            REQUIRE_EQ(PLDSIZE, size2);
            REQUIRE_EQ(size1 + size2, out_stream.byte_count());
            // back_up a size that's valid for all ZeroCopyOutputStream
            out_stream.back_up(PLDSIZE / 2);
            REQUIRE_EQ(size1 + size2 - PLDSIZE / 2, out_stream.byte_count());
            void* blk3 = nullptr;
            int size3 = 0;
            REQUIRE(out_stream.next(&blk3, &size3));
            REQUIRE_EQ((char*)blk2 + PLDSIZE / 2, blk3);
            REQUIRE_EQ(PLDSIZE / 2, size3);
            REQUIRE_EQ(size1 + size2, out_stream.byte_count());

            // back_up a size that's undefined in regular ZeroCopyOutputStream
            out_stream.back_up(PLDSIZE * 2);
            REQUIRE_EQ(0, out_stream.byte_count());
            void* blk4 = nullptr;
            int size4 = 0;
            REQUIRE(out_stream.next(&blk4, &size4));
            REQUIRE_EQ(PLDSIZE, size4);
            REQUIRE_EQ(size4, out_stream.byte_count());
            if (i == 1) {
                REQUIRE_EQ(blk1, blk4);
            }
            void* blk5 = nullptr;
            int size5 = 0;
            REQUIRE(out_stream.next(&blk5, &size5));
            REQUIRE_EQ(PLDSIZE, size5);
            REQUIRE_EQ(size4 + size5, out_stream.byte_count());
            if (i == 1) {
                REQUIRE_EQ(blk2, blk5);
            }
        }
    }

    TEST_CASE_FIXTURE(IOBufTest, "backup_iobuf_never_called_next") {
        {
            // Consume the left TLS block so that later cases are easier
            // to check.
            turbo::IOBuf dummy;
            turbo::IOBufAsZeroCopyOutputStream dummy_stream(&dummy);
            void* dummy_data = nullptr;
            int dummy_size = 0;
            REQUIRE(dummy_stream.next(&dummy_data, &dummy_size));
        }
        turbo::IOBuf src;
        const size_t N = DEFAULT_PAYLOAD * 2;
        src.resize(N);
        REQUIRE_EQ(2u, src.backing_block_num());
        REQUIRE_EQ(N, src.size());
        turbo::IOBufAsZeroCopyOutputStream out_stream(&src);
        out_stream.back_up(1); // also succeed.
        REQUIRE_EQ(-1, out_stream.byte_count());
        REQUIRE_EQ(DEFAULT_PAYLOAD * 2 - 1, src.size());
        REQUIRE_EQ(2u, src.backing_block_num());
        void* data0 = nullptr;
        int size0 = 0;
        REQUIRE(out_stream.next(&data0, &size0));
        REQUIRE_EQ(1, size0);
        REQUIRE_EQ(0, out_stream.byte_count());
        REQUIRE_EQ(2u, src.backing_block_num());
        void* data1 = nullptr;
        int size1 = 0;
        REQUIRE(out_stream.next(&data1, &size1));
        REQUIRE_EQ(size1, out_stream.byte_count());
        REQUIRE_EQ(3u, src.backing_block_num());
        REQUIRE_EQ(N + size1, src.size());
        void* data2 = nullptr;
        int size2 = 0;
        REQUIRE(out_stream.next(&data2, &size2));
        REQUIRE_EQ(size1 + size2, out_stream.byte_count());
        REQUIRE_EQ(4u, src.backing_block_num());
        REQUIRE_EQ(N + size1 + size2, src.size());
        TLOG_INFO("Backup1");
        out_stream.back_up(size1); // intended size1, not size2 to make this back_up
        // to cross boundary of blocks.
        REQUIRE_EQ(size2, out_stream.byte_count());
        REQUIRE_EQ(N + size2, src.size());
        TLOG_INFO("Backup2");
        out_stream.back_up(size2);
        REQUIRE_EQ(0, out_stream.byte_count());
        REQUIRE_EQ(N, src.size());
    }

    void *backup_thread(void *arg) {
        turbo::IOBufAsZeroCopyOutputStream *wrapper =
                (turbo::IOBufAsZeroCopyOutputStream *)arg;
        wrapper->back_up(1024);
        return nullptr;
    }

    TEST_CASE_FIXTURE(IOBufTest, "backup_in_another_thread") {
        turbo::IOBuf buf;
        turbo::IOBufAsZeroCopyOutputStream wrapper(&buf);
        size_t alloc_size = 0;
        for (int i = 0; i < 10; ++i) {
            void *data;
            int len;
            REQUIRE(wrapper.next(&data, &len));
            alloc_size += len;
        }
        REQUIRE_EQ(alloc_size, buf.length());
        for (int i = 0; i < 10; ++i) {
            void *data;
            int len;
            REQUIRE(wrapper.next(&data, &len));
            alloc_size += len;
            pthread_t tid;
            pthread_create(&tid, nullptr, backup_thread, &wrapper);
            pthread_join(tid, nullptr);
        }
        REQUIRE_EQ(alloc_size - 1024 * 10, buf.length());
    }

    TEST_CASE_FIXTURE(IOBufTest, "own_block") {
        turbo::IOBuf buf;
        const ssize_t BLOCK_SIZE = 1024;
        turbo::IOBuf::Block *saved_tls_block = turbo::files_internal::iobuf::get_tls_block_head();
        turbo::IOBufAsZeroCopyOutputStream wrapper(&buf, BLOCK_SIZE);
        int alloc_size = 0;
        for (int i = 0; i < 100; ++i) {
            void *data;
            int size;
            REQUIRE(wrapper.next(&data, &size));
            alloc_size += size;
            if (size > 1) {
                wrapper.back_up(1);
                alloc_size -= 1;
            }
        }
        REQUIRE_EQ(static_cast<size_t>(alloc_size), buf.length());
        REQUIRE_EQ(saved_tls_block, turbo::files_internal::iobuf::get_tls_block_head());
        REQUIRE_EQ(turbo::files_internal::iobuf::block_cap(buf._front_ref().block), BLOCK_SIZE - BLOCK_OVERHEAD);
    }

    struct Foo1 {
        explicit Foo1(int x2) : x(x2) {}
        int x;
    };

    struct Foo2 {
        std::vector<Foo1> foo1;
    };

    std::ostream& operator<<(std::ostream& os, const Foo1& foo1) {
        return os << "foo1.x=" << foo1.x;
    }

    std::ostream& operator<<(std::ostream& os, const Foo2& foo2) {
        for (size_t i = 0; i < foo2.foo1.size(); ++i) {
            os << "foo2[" << i << "]=" << foo2.foo1[i];
        }
        return os;
    }

    TEST_CASE_FIXTURE(IOBufTest, "as_ostream") {
        turbo::files_internal::iobuf::reset_blockmem_allocate_and_deallocate();

        turbo::IOBufBuilder builder;
        TLOG_INFO("sizeof(IOBufBuilder)={}\n sizeof(IOBuf)={}\n sizeof(IOBufAsZeroCopyOutputStream)={}\n sizeof(ZeroCopyStreamAsStreamBuf)={}\n sizeof(ostream)={}",
                  sizeof(builder), sizeof(turbo::IOBuf),
                  sizeof(turbo::IOBufAsZeroCopyOutputStream),
                  sizeof(turbo::ZeroCopyStreamAsStreamBuf),
                  sizeof(std::ostream));
        int x = -1;
        builder << 2 << " " << x << " " << 1.1 << " hello ";
        REQUIRE_EQ("2 -1 1.1 hello ", builder.buf().to_string());

        builder << "world!";
        REQUIRE_EQ("2 -1 1.1 hello world!", builder.buf().to_string());

        builder.buf().clear();
        builder << "world!";
        REQUIRE_EQ("world!", builder.buf().to_string());

        Foo2 foo2;
        for (int i = 0; i < 10000; ++i) {
            foo2.foo1.push_back(Foo1(i));
        }
        builder.buf().clear();
        builder << "<before>" << foo2 << "<after>";
        std::ostringstream oss;
        oss << "<before>" << foo2 << "<after>";
        REQUIRE_EQ(oss.str(), builder.buf().to_string());

        turbo::IOBuf target;
        builder.move_to(target);
        REQUIRE(builder.buf().empty());
        REQUIRE_EQ(oss.str(), target.to_string());

        std::ostringstream oss2;
        oss2 << target;
        REQUIRE_EQ(oss.str(), oss2.str());
    }

    TEST_CASE_FIXTURE(IOBufTest, "append_from_fd_with_offset") {
        turbo::TempFile file;
        REQUIRE(file.open().ok());
        REQUIRE(file.write("dummy").ok());
        turbo::FDGuard fd(open(file.path().c_str(), O_RDWR | O_TRUNC));
        REQUIRE_GE(fd , 0);
        turbo::IOPortal buf;
        char dummy[10 * 1024];
        buf.append(dummy, sizeof(dummy));
        REQUIRE_EQ((ssize_t)sizeof(dummy), buf.cut_into_file_descriptor(fd).value());
        for (size_t i = 0; i < sizeof(dummy); ++i) {
            turbo::IOPortal b0;
            REQUIRE_EQ(sizeof(dummy) - i, (size_t)b0.pappend_from_file_descriptor(fd, i, sizeof(dummy)).value_or(-1)) ;
            char tmp[sizeof(dummy)];
            REQUIRE_EQ(0, memcmp(dummy + i, b0.fetch(tmp, b0.length()), b0.length()));
        }

    }

    static std::atomic<int> s_nthread(0);
    static long number_per_thread = 1024;

    void* cut_into_fd(void* arg) {
        int fd = (int)(long)arg;
        const long start_num = number_per_thread *
                               s_nthread.fetch_add(1, std::memory_order_relaxed);
        off_t offset = start_num * sizeof(int);
        for (int i = 0; i < number_per_thread; ++i) {
            int to_write = start_num + i;
            turbo::IOBuf out;
            out.append(&to_write, sizeof(int));
            CHECK_EQ(out.pcut_into_file_descriptor(fd, offset + sizeof(int) * i).value_or(-1),
                     (ssize_t)sizeof(int));
        }
        return nullptr;
    }

    TEST_CASE_FIXTURE(IOBufTest, "cut_into_fd_with_offset_multithreaded") {
        s_nthread.store(0);
        number_per_thread = 10240;
        pthread_t threads[8];
        long fd = open(".out.txt", O_RDWR | O_CREAT | O_TRUNC, 0644);
        REQUIRE_GE(fd , 0);
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(threads); ++i) {
            REQUIRE_EQ(0, pthread_create(&threads[i], nullptr, cut_into_fd, (void*)fd));
        }
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(threads); ++i) {
            pthread_join(threads[i], nullptr);
        }
        for (int i = 0; i < number_per_thread * (int)TURBO_ARRAY_SIZE(threads); ++i) {
            off_t offset = i * sizeof(int);
            turbo::IOPortal in;
            REQUIRE_EQ((ssize_t)sizeof(int), in.pappend_from_file_descriptor(fd, offset, sizeof(int)).value_or(-1));
            int j;
            REQUIRE_EQ(sizeof(j), in.cutn(&j, sizeof(j)));
            REQUIRE_EQ(i, j);
        }
    }

    TEST_CASE_FIXTURE(IOBufTest, "slice") {
        size_t N = 100000;
        std::string expected;
        expected.reserve(N);
        turbo::IOBuf buf;
        for (size_t i = 0; i < N; ++i) {
            expected.push_back(i % 26 + 'a');
            buf.push_back(i % 26 + 'a');
        }
        const size_t block_count = buf.backing_block_num();
        std::string actual;
        actual.reserve(expected.size());
        for (size_t i = 0; i < block_count; ++i) {
            std::string_view p = buf.backing_block(i);
            REQUIRE_FALSE(p.empty());
            actual.append(p.data(), p.size());
        }
        REQUIRE_EQ(expected , actual);
    }

    TEST_CASE_FIXTURE(IOBufTest, "swap") {
        turbo::IOBuf a;
        a.append("I'am a");
        turbo::IOBuf b;
        b.append("I'am b");
        std::swap(a, b);
        REQUIRE(a.equals("I'am b"));
        REQUIRE(b.equals("I'am a"));
    }

    TEST_CASE_FIXTURE(IOBufTest, "resize") {
        turbo::IOBuf a;
        a.resize(100);
        std::string as;
        as.resize(100);
        turbo::IOBuf b;
        b.resize(100, 'b');
        std::string bs;
        bs.resize(100, 'b');
        REQUIRE_EQ(100u, a.length());
        REQUIRE_EQ(100u, b.length());
        REQUIRE(a.equals(as));
        REQUIRE(b.equals(bs));
    }

    TEST_CASE_FIXTURE(IOBufTest, "iterate_bytes") {
        turbo::IOBuf a;
        a.append("hello world");
        std::string saved_a = a.to_string();
        size_t n = 0;
        turbo::IOBufBytesIterator it(a);
        for (; it != nullptr; ++it, ++n) {
            REQUIRE_EQ(saved_a[n], *it);
        }
        REQUIRE_EQ(saved_a.size(), n);
        REQUIRE_EQ(saved_a , a);

        // append more to the iobuf, iterator should still be ended.
        a.append(", this is iobuf");
        REQUIRE_EQ(it , nullptr);

        // append more-than-one-block data to the iobuf
        for (int i = 0; i < 1024; ++i) {
            a.append("ab33jm4hgaklkv;2[25lj4lkj312kl4j321kl4j3k");
        }
        saved_a = a.to_string();
        n = 0;
        for (turbo::IOBufBytesIterator it2(a); it2 != nullptr; it2++/*intended post++*/, ++n) {
            REQUIRE_EQ(saved_a[n], *it2);
        }
        REQUIRE_EQ(saved_a.size(), n);
        REQUIRE_EQ(saved_a, a);
    }

    TEST_CASE_FIXTURE(IOBufTest, "appender") {
        turbo::IOBufAppender appender;
        REQUIRE_EQ(0, appender.append("hello", 5));
        REQUIRE_EQ("hello", appender.buf());
        REQUIRE_EQ(0, appender.push_back(' '));
        REQUIRE_EQ(0, appender.append("world", 5));
        REQUIRE_EQ("hello world", appender.buf());
        turbo::IOBuf buf2;
        appender.move_to(buf2);
        REQUIRE_EQ("", appender.buf());
        REQUIRE_EQ("hello world", buf2);
        std::string str;
        for (int i = 0; i < 10000; ++i) {
            char buf[128];
            int len = snprintf(buf, sizeof(buf), "1%d2%d3%d4%d5%d", i, i, i, i, i);
            appender.append(buf, len);
            str.append(buf, len);
        }
        REQUIRE_EQ(str, appender.buf());
        turbo::IOBuf buf3;
        appender.move_to(buf3);
        REQUIRE_EQ("", appender.buf());
        REQUIRE_EQ(str, buf3);
    }

    TEST_CASE_FIXTURE(IOBufTest, "appender_perf") {
        const size_t N1 = 100000;
        turbo::StopWatcher tm1;
        tm1.reset();
        turbo::IOBuf buf1;
        for (size_t i = 0; i < N1; ++i) {
            buf1.push_back(i);
        }
        tm1.stop();

        turbo::StopWatcher tm2;
        tm2.reset();
        turbo::IOBufAppender appender1;
        for (size_t i = 0; i < N1; ++i) {
            appender1.push_back(i);
        }
        tm2.stop();

        TLOG_INFO("IOBuf.push_back={}ns IOBufAppender.push_back={}ns",
                  tm1.elapsed_nano() / N1, tm2.elapsed_nano() / N1);

        const size_t N2 = 50000;
        const std::string s = "a repeatly appended string";
        std::string str2;
        turbo::IOBuf buf2;
        tm1.reset();
        for (size_t i = 0; i < N2; ++i) {
            buf2.append(s);
        }
        tm1.stop();

        tm2.reset();
        turbo::IOBufAppender appender2;
        for (size_t i = 0; i < N2; ++i) {
            appender2.append(s);
        }
        tm2.stop();

        turbo::StopWatcher tm3;
        tm3.reset();
        for (size_t i = 0; i < N2; ++i) {
            str2.append(s);
        }
        tm3.stop();

        TLOG_INFO("IOBuf.push_back={}ns IOBufAppender.push_back={}ns IOBuf.append={}ns IOBufAppender.append={}ns string.append={}ns",
                  tm1.elapsed_nano() / N1, tm2.elapsed_nano() / N1,
                  tm1.elapsed_nano() / N2, tm2.elapsed_nano() / N2,
                  tm3.elapsed_nano() / N2);
    }

    TEST_CASE_FIXTURE(IOBufTest, "printed_as_binary") {
        turbo::IOBuf buf;
        std::string str;
        for (int i = 0; i < 256; ++i) {
            buf.push_back((char)i);
            str.push_back((char)i);
        }
        const char* const OUTPUT =
                "\\00\\01\\02\\03\\04\\05\\06\\07\\b\\t\\n\\0B\\0C\\r\\0E\\0F"
                "\\10\\11\\12\\13\\14\\15\\16\\17\\18\\19\\1A\\1B\\1C\\1D\\1E"
                "\\1F !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUV"
                "WXYZ[\\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\\7F\\80\\81\\82"
                "\\83\\84\\85\\86\\87\\88\\89\\8A\\8B\\8C\\8D\\8E\\8F\\90\\91"
                "\\92\\93\\94\\95\\96\\97\\98\\99\\9A\\9B\\9C\\9D\\9E\\9F\\A0"
                "\\A1\\A2\\A3\\A4\\A5\\A6\\A7\\A8\\A9\\AA\\AB\\AC\\AD\\AE\\AF"
                "\\B0\\B1\\B2\\B3\\B4\\B5\\B6\\B7\\B8\\B9\\BA\\BB\\BC\\BD\\BE"
                "\\BF\\C0\\C1\\C2\\C3\\C4\\C5\\C6\\C7\\C8\\C9\\CA\\CB\\CC\\CD"
                "\\CE\\CF\\D0\\D1\\D2\\D3\\D4\\D5\\D6\\D7\\D8\\D9\\DA\\DB\\DC"
                "\\DD\\DE\\DF\\E0\\E1\\E2\\E3\\E4\\E5\\E6\\E7\\E8\\E9\\EA\\EB"
                "\\EC\\ED\\EE\\EF\\F0\\F1\\F2\\F3\\F4\\F5\\F6\\F7\\F8\\F9\\FA"
                "\\FB\\FC\\FD\\FE\\FF";
        std::ostringstream os;
        os << turbo::ToPrintable(buf, 256);
        REQUIRE_EQ(OUTPUT, os.str());
        os.str("");
        os << turbo::ToPrintable(str, 256);
        REQUIRE_EQ(OUTPUT, os.str());
        auto fstr = turbo::format("{}", turbo::ToPrintable(buf, 256));
        REQUIRE_EQ(fstr, os.str());
    }

    TEST_CASE_FIXTURE(IOBufTest, "copy_to_string_from_iterator") {
        turbo::IOBuf b0;
        for (size_t i = 0; i < 1 * 1024 * 1024lu; ++i) {
            b0.push_back(turbo::fast_uniform('a', 'z'));
        }
        turbo::IOBuf b1(b0);
        turbo::IOBufBytesIterator iter(b0);
        size_t nc = 0;
        while (nc < b0.length()) {
            size_t to_copy = turbo::fast_uniform(1024lu, 64 * 1024lu);
            turbo::IOBuf b;
            b1.cutn(&b, to_copy);
            std::string s;
            const size_t copied = iter.copy_and_forward(&s, to_copy);
            REQUIRE_GT(copied, 0u);
            REQUIRE(b.equals(s));
            nc += copied;
        }
        REQUIRE_EQ(nc, b0.length());
    }

    static void* my_free_params = nullptr;
    static void my_free(void* m) {
        free(m);
        my_free_params = m;
    }

    TEST_CASE_FIXTURE(IOBufTest, "append_user_data_and_consume") {
        turbo::IOBuf b0;
        const int REP = 16;
        const size_t len = REP * 256;
        char* data = (char*)malloc(len);
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < REP; ++j) {
                data[i * REP + j] = (char)i;
            }
        }
        my_free_params = nullptr;
        REQUIRE_EQ(0, b0.append_user_data(data, len, my_free));
        REQUIRE_EQ(1UL, b0._ref_num());
        turbo::IOBuf::BlockRef r = b0._front_ref();
        REQUIRE_EQ(1, turbo::files_internal::iobuf::block_shared_count(r.block));
        REQUIRE_EQ(len, b0.size());
        std::string out;
        REQUIRE_EQ(len, b0.cutn(&out, len));
        REQUIRE(b0.empty());
        REQUIRE_EQ(data, my_free_params);

        REQUIRE_EQ(len, out.size());
        // note: cannot memcmp with data which is already free-ed
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < REP; ++j) {
                REQUIRE_EQ((char)i, out[i * REP + j]);
            }
        }
    }

    TEST_CASE_FIXTURE(IOBufTest, "append_user_data_and_share") {
        turbo::IOBuf b0;
        const int REP = 16;
        const size_t len = REP * 256;
        char* data = (char*)malloc(len);
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < REP; ++j) {
                data[i * REP + j] = (char)i;
            }
        }
        my_free_params = nullptr;
        REQUIRE_EQ(0, b0.append_user_data(data, len, my_free));
        REQUIRE_EQ(1UL, b0._ref_num());
        turbo::IOBuf::BlockRef r = b0._front_ref();
        REQUIRE_EQ(1, turbo::files_internal::iobuf::block_shared_count(r.block));
        REQUIRE_EQ(len, b0.size());

        {
            turbo::IOBuf bufs[256];
            for (int i = 0; i < 256; ++i) {
                REQUIRE_EQ((size_t)REP, b0.cutn(&bufs[i], REP));
                REQUIRE_EQ(len - (i+1) * REP, b0.size());
                if (i != 255) {
                    REQUIRE_EQ(1UL, b0._ref_num());
                    turbo::IOBuf::BlockRef r = b0._front_ref();
                    REQUIRE_EQ(i + 2, turbo::files_internal::iobuf::block_shared_count(r.block));
                } else {
                    REQUIRE_EQ(0UL, b0._ref_num());
                    REQUIRE(b0.empty());
                }
            }
            REQUIRE_EQ(nullptr, my_free_params);
            for (int i = 0; i < 256; ++i) {
                std::string out = bufs[i].to_string();
                REQUIRE_EQ((size_t)REP, out.size());
                for (int j = 0; j < REP; ++j) {
                    REQUIRE_EQ((char)i, out[j]);
                }
            }
        }
        REQUIRE_EQ(data, my_free_params);
    }

    TEST_CASE_FIXTURE(IOBufTest, "append_user_data_with_meta") {
        turbo::IOBuf b0;
        const int REP = 16;
        const size_t len = 256;
        char* data[REP];
        for (int i = 0; i < REP; ++i) {
            data[i] = (char*)malloc(len);
            REQUIRE_EQ(0, b0.append_user_data_with_meta(data[i], len, my_free, i));
        }
        for (int i = 0; i < REP; ++i) {
            REQUIRE_EQ(i, b0.get_first_data_meta());
            turbo::IOBuf out;
            REQUIRE_EQ(len / 2, b0.cutn(&out, len / 2));
            REQUIRE_EQ(i, b0.get_first_data_meta());
            REQUIRE_EQ(len / 2, b0.cutn(&out, len / 2));
        }
    }

    TEST_CASE_FIXTURE(IOBufTest, "share_tls_block") {
        turbo::files_internal::iobuf::remove_tls_block_chain();
        turbo::IOBuf::Block* b = turbo::files_internal::iobuf::acquire_tls_block();
        REQUIRE_EQ(0u, turbo::files_internal::iobuf::block_size(b));

        turbo::IOBuf::Block* b2 = turbo::files_internal::iobuf::share_tls_block();
        turbo::IOBuf buf;
        for (size_t i = 0; i < turbo::files_internal::iobuf::block_cap(b2); i++) {
            buf.push_back('x');
        }
        // after pushing to b2, b2 is full but it is still head of tls block.
        REQUIRE_NE(b, b2);
        turbo::files_internal::iobuf::release_tls_block_chain(b);
        REQUIRE_EQ(b, turbo::files_internal::iobuf::share_tls_block());
        // After releasing b, now tls block is b(not full) -> b2(full) -> nullptr
        for (size_t i = 0; i < turbo::files_internal::iobuf::block_cap(b); i++) {
            buf.push_back('x');
        }
        // now tls block is b(full) -> b2(full) -> nullptr
        turbo::IOBuf::Block* head_block = turbo::files_internal::iobuf::share_tls_block();
        REQUIRE_EQ(0u, turbo::files_internal::iobuf::block_size(head_block));
        REQUIRE_NE(b, head_block);
        REQUIRE_NE(b2, head_block);
    }

    TEST_CASE_FIXTURE(IOBufTest, "acquire_tls_block") {
        turbo::files_internal::iobuf::remove_tls_block_chain();
        turbo::IOBuf::Block* b = turbo::files_internal::iobuf::acquire_tls_block();
        const size_t block_cap = turbo::files_internal::iobuf::block_cap(b);
        turbo::IOBuf buf;
        for (size_t i = 0; i < block_cap; i++) {
            buf.append("x");
        }
        REQUIRE_EQ(1, turbo::files_internal::iobuf::get_tls_block_count());
        turbo::IOBuf::Block* head = turbo::files_internal::iobuf::get_tls_block_head();
        REQUIRE_EQ(turbo::files_internal::iobuf::block_cap(head), turbo::files_internal::iobuf::block_size(head));
        turbo::files_internal::iobuf::release_tls_block_chain(b);
        REQUIRE_EQ(2, turbo::files_internal::iobuf::get_tls_block_count());
        for (size_t i = 0; i < block_cap; i++) {
            buf.append("x");
        }
        REQUIRE_EQ(2, turbo::files_internal::iobuf::get_tls_block_count());
        head = turbo::files_internal::iobuf::get_tls_block_head();
        REQUIRE_EQ(turbo::files_internal::iobuf::block_cap(head), turbo::files_internal::iobuf::block_size(head));
        b = turbo::files_internal::iobuf::acquire_tls_block();
        REQUIRE_EQ(0, turbo::files_internal::iobuf::get_tls_block_count());
        REQUIRE_NE(turbo::files_internal::iobuf::block_cap(b), turbo::files_internal::iobuf::block_size(b));
    }

} // namespace
