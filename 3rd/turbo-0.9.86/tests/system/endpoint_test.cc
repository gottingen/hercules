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
// Created by jeff on 24-1-5.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/system/endpoint.h"
#include "turbo/system/internal/endpoint_internal.h"
#include "turbo/log/logging.h"
#include "turbo/container/flat_hash_map.h"

namespace {

    using turbo::system_internal::ExtendedEndPoint;

    TEST_CASE("EndPointTest, comparisons") {
        turbo::EndPoint p1(1234, 5678);
        turbo::EndPoint p2 = p1;
        REQUIRE((p1 == p2 && !(p1 != p2)));
        REQUIRE((p1 <= p2 && p1 >= p2 && !(p1 < p2 || p1 > p2)));
        p2.set(p1.ip(), p1.port()+1);
        REQUIRE((p1 != p2 && !(p1 == p2)));
        REQUIRE((p1 < p2 && p2 > p1 && !(p2 <= p1 || p1 >= p2)));
        p2.set(p2.ip().num()-1, p1.port());
        REQUIRE((p1 != p2 && !(p1 == p2)));
        REQUIRE((p1 > p2 && p2 < p1 && !(p1 <= p2 || p2 >= p1)));
    }

    TEST_CASE("EndPointTest, ip_t") {
        TLOG_INFO("INET6_ADDRSTRLEN = {}", INET6_ADDRSTRLEN);

        turbo::IPAddr ip0;
        REQUIRE_EQ(true, ip0.parse("1.1.1.1"));
        REQUIRE_EQ("1.1.1.1", ip0.to_string());
        REQUIRE_EQ(false, ip0.parse("301.1.1.1"));
        REQUIRE_EQ(false, ip0.parse("1.-1.1.1"));
        REQUIRE_EQ(false, ip0.parse("1.1.-101.1"));
        REQUIRE_EQ("1.0.0.0", turbo::IPAddr(1).to_string());

        turbo::IPAddr ip1, ip2, ip3;
        REQUIRE_EQ(true, ip1.parse("192.168.0.1"));
        REQUIRE_EQ(true, ip2.parse("192.168.0.2"));
        ip3 = ip1;
        REQUIRE_LT(ip1, ip2);
        REQUIRE_LE(ip1, ip2);
        REQUIRE_GT(ip2, ip1);
        REQUIRE_GE(ip2, ip1);
        REQUIRE(ip1 != ip2);
        REQUIRE_FALSE(ip1 == ip2);
        REQUIRE(ip1 == ip3);
        REQUIRE_FALSE(ip1 != ip3);
    }

    TEST_CASE("EndPointTest, show_local_info") {
        //TLOG_INFO("my_ip is {} my_ip_cstr is {} my_hostname is {}", turbo::int2ip(turbo::my_ip()), turbo::my_ip_cstr(),
         //         turbo::my_hostname());
    }

    TEST_CASE("EndPointTest, endpoint") {
        turbo::EndPoint p1;
        REQUIRE_EQ(turbo::IPAddr::any(), p1.ip());
        REQUIRE_EQ(0, p1.port());

        turbo::EndPoint p2(turbo::IPAddr::none(), -1);
        REQUIRE_EQ(turbo::IPAddr::none(), p2.ip());
        REQUIRE_EQ(-1, p2.port());

        turbo::EndPoint p3;
        REQUIRE_EQ(false, p3.parse(" 127.0.0.1:-1"));
        REQUIRE_EQ(false, p3.parse(" 127.0.0.1:65536"));
        REQUIRE_EQ(true, p3.parse(" 127.0.0.1:65535"));
        REQUIRE_EQ(true, p3.parse(" 127.0.0.1:0"));

        turbo::EndPoint p4;
        REQUIRE_EQ(true, p4.parse(" 127.0.0.1: 289 "));
        REQUIRE_EQ("127.0.0.1", p4.ip().to_string());
        REQUIRE_EQ(289, p4.port());

        turbo::EndPoint p5;
        REQUIRE_EQ(false, p5.parse_hostname("localhost:-1"));
        REQUIRE_EQ(false, p5.parse_hostname("localhost:65536"));
        REQUIRE_EQ(true, p5.parse_hostname("localhost:65535"));
        REQUIRE_EQ(true, p5.parse_hostname("localhost:0"));

    }

    TEST_CASE("EndPointTest, hash_table") {
        turbo::flat_hash_map<turbo::EndPoint, int> m;
        turbo::EndPoint ep1(123);
        turbo::EndPoint ep2(turbo::IPAddr::any(), 456);
        ++m[ep1];
        REQUIRE(m.find(ep1) != m.end());
        REQUIRE_EQ(1, m.find(ep1)->second);
        REQUIRE_EQ(1u, m.size());

        ++m[ep1];
        REQUIRE(m.find(ep1) != m.end());
        REQUIRE_EQ(2, m.find(ep1)->second);
        REQUIRE_EQ(1u, m.size());

        ++m[ep2];
        REQUIRE(m.find(ep2) != m.end());
        REQUIRE_EQ(1, m.find(ep2)->second);
        REQUIRE_EQ(2u, m.size());
    }

    TEST_CASE("EndPointTest, flat_map") {
        turbo::flat_hash_map<turbo::EndPoint, int> m;
        uint32_t port = 8088;

        turbo::EndPoint ep1( port);
        turbo::EndPoint ep2( port);
        ++m[ep1];
        ++m[ep2];
        REQUIRE_EQ(1u, m.size());

        turbo::IPAddr ip_addr;
        ip_addr.parse("10.10.10.10");
        int ip = ip_addr.num();

        for (int i = 0; i < 1023; ++i) {
            turbo::EndPoint ep(turbo::IPAddr(++ip), port);
            ++m[ep];
        }
    }

    void* server_proc(void* arg) {
        int listen_fd = (int64_t)arg;
        sockaddr_storage ss;
        socklen_t len = sizeof(ss);
        int fd = accept(listen_fd, (sockaddr*)&ss, &len);
        return (void*)(int64_t)fd;
    }

    static void test_listen_connect(const std::string& server_addr, const std::string& exp_client_addr) {
        turbo::EndPoint point;
        REQUIRE_EQ(true, point.parse(server_addr.c_str()));
        REQUIRE_EQ(server_addr, point.to_string());

        auto listen_fd = point.tcp_listen();
        turbo::println("listen_fd = {}", listen_fd.status().to_string());
        REQUIRE(listen_fd.ok());
        REQUIRE_GT(listen_fd.value(), 0);
        pthread_t pid;
        pthread_create(&pid, nullptr, server_proc, (void*)(int64_t)listen_fd.value());

        auto fd = point.tcp_connect(nullptr);
        REQUIRE(fd.ok());
        REQUIRE_GT(fd.value(), 0);

        turbo::EndPoint point2;
        REQUIRE_EQ(true, point2.local_side(fd.value()));

        std::string s = point2.to_string();
        if (point2.get_endpoint_type() == AF_UNIX) {
            REQUIRE_EQ(exp_client_addr, s);
        } else {
            REQUIRE_EQ(exp_client_addr, s.substr(0, exp_client_addr.size()));
        }
        REQUIRE_EQ(true, point2.remote_side(fd.value()));
        REQUIRE_EQ(server_addr, point2.to_string());
        close(fd.value());

        void* ret = nullptr;
        pthread_join(pid, &ret);
        int server_fd = (int)(int64_t)ret;
        REQUIRE_GT(server_fd, 0);
        close(server_fd);
        close(listen_fd.value());
    }

    static void test_parse_and_serialize(const std::string& instr, const std::string& outstr) {
        turbo::EndPoint ep;
        REQUIRE(ep.parse(instr.c_str()));
        turbo::EndPointStr s = ep.to_str();
        REQUIRE_EQ(outstr, std::string(s.c_str()));
    }

    TEST_CASE("EndPointTest, ipv4") {
        test_listen_connect("127.0.0.1:8787", "127.0.0.1:");
    }

    TEST_CASE("EndPointTest, ipv6") {
        // FIXME: test environ may not support ipv6
        // test_listen_connect("[::1]:8787", "[::1]:");

        test_parse_and_serialize("[::1]:8080", "[::1]:8080");
        test_parse_and_serialize("  [::1]:65535  ", "[::1]:65535");
        test_parse_and_serialize("  [2001:0db8:a001:0002:0003:0ab9:C0A8:0102]:65535  ",
                                 "[2001:db8:a001:2:3:ab9:c0a8:102]:65535");

        turbo::EndPoint ep;
        REQUIRE_EQ(false, ep.parse("[2001:db8:1:2:3:ab9:c0a8:102]"));
        REQUIRE_EQ(false, ep.parse("[2001:db8:1:2:3:ab9:c0a8:102]#654321"));
        REQUIRE_EQ(false, ep.parse("ipv6:2001:db8:1:2:3:ab9:c0a8:102"));
        REQUIRE_EQ(false, ep.parse("["));
        REQUIRE_EQ(false, ep.parse("[::1"));
        REQUIRE_EQ(false, ep.parse("[]:80"));
        REQUIRE_EQ(false, ep.parse("[]"));
        REQUIRE_EQ(false, ep.parse("[]:"));
    }

    TEST_CASE("EndPointTest, unix_socket") {
        ::unlink("test.sock");
        test_listen_connect("unix:test.sock", "unix:");
        ::unlink("test.sock");

        turbo::EndPoint point;
        REQUIRE_EQ(false, point.parse(""));
        REQUIRE_EQ(false, point.parse("a.sock"));
        REQUIRE_EQ(false, point.parse("unix:"));
        REQUIRE_EQ(false, point.parse(" unix: "));
        REQUIRE_EQ(true, point.parse("unix://a.sock", 123));
        REQUIRE_EQ(std::string("unix://a.sock"), point.to_string());

        std::string long_path = "unix:";
        long_path.append(sizeof(sockaddr_un::sun_path) - 1, 'a');
        REQUIRE_EQ(true, point.parse(long_path.c_str()));
        REQUIRE_EQ(long_path, point.to_string());
        long_path.push_back('a');
        REQUIRE_EQ(false, point.parse(long_path.c_str()));
        char buf[128] = {0}; // braft use this size of buffer
        size_t ret = snprintf(buf, sizeof(buf), "%s:%d", point.to_str().c_str(), INT_MAX);
        REQUIRE_LT(ret, sizeof(buf) - 1);
    }

    TEST_CASE("EndPointTest, original_endpoint") {
        turbo::EndPoint ep;
        REQUIRE_FALSE(ExtendedEndPoint::is_extended(ep));
        REQUIRE_EQ(nullptr, ExtendedEndPoint::address(ep));

        REQUIRE_EQ(true, ep.parse("1.2.3.4:5678"));
        REQUIRE_FALSE(ExtendedEndPoint::is_extended(ep));
        REQUIRE_EQ(nullptr, ExtendedEndPoint::address(ep));

        // ctor & dtor
        {
            turbo::EndPoint ep2(ep);
            REQUIRE_FALSE(ExtendedEndPoint::is_extended(ep));
            REQUIRE_EQ(ep.ip(), ep2.ip());
            REQUIRE_EQ(ep.port(), ep2.port());
        }

        // assign
        turbo::EndPoint ep2;
        ep2 = ep;
        REQUIRE_EQ(ep.ip(), ep2.ip());
        REQUIRE_EQ(ep.port(), ep2.port());
    }

    TEST_CASE("EndPointTest, extended_endpoint") {
        turbo::EndPoint ep;
        REQUIRE_EQ(true, ep.parse("unix:sock.file"));
        REQUIRE(ExtendedEndPoint::is_extended(ep));
        ExtendedEndPoint* eep = ExtendedEndPoint::address(ep);
        REQUIRE(eep);
        REQUIRE_EQ(AF_UNIX, eep->family());
        REQUIRE_EQ(1, eep->_ref_count.load());

        // copy ctor & dtor
        {
            turbo::EndPoint tmp(ep);
            REQUIRE_EQ(2, eep->_ref_count.load());
            REQUIRE_EQ(eep, ExtendedEndPoint::address(tmp));
            REQUIRE_EQ(eep, ExtendedEndPoint::address(ep));
        }
        REQUIRE_EQ(1, eep->_ref_count.load());

        turbo::EndPoint ep2;

        // extended endpoint assigns to original endpoint
        ep2 = ep;
        REQUIRE_EQ(2, eep->_ref_count.load());
        REQUIRE_EQ(eep, ExtendedEndPoint::address(ep2));

        // original endpoint assigns to extended endpoint
        ep2 = turbo::EndPoint();
        REQUIRE_EQ(1, eep->_ref_count.load());
        REQUIRE_FALSE(ExtendedEndPoint::is_extended(ep2));

        // extended endpoint assigns to extended endpoint
        REQUIRE_EQ(true, ep2.parse("[::1]:2233"));
        ExtendedEndPoint* eep2 = ExtendedEndPoint::address(ep2);
        REQUIRE(eep2);
        ep2 = ep;
        // eep2 has been returned to resource pool, but we can still access it here unsafely.
        REQUIRE_EQ(0, eep2->_ref_count.load());
        REQUIRE_EQ(AF_UNSPEC, eep2->family());
        REQUIRE_EQ(2, eep->_ref_count.load());
        REQUIRE_EQ(eep, ExtendedEndPoint::address(ep));
        REQUIRE_EQ(eep, ExtendedEndPoint::address(ep2));

        REQUIRE_EQ(true, ep2.parse("[::1]:2233"));
        REQUIRE_EQ(1, eep->_ref_count.load());
        eep2 = ExtendedEndPoint::address(ep2);
        REQUIRE_NE(eep, eep2);
        REQUIRE_EQ(1, eep2->_ref_count.load());
    }

    TEST_CASE("EndPointTest, endpoint_compare") {
        turbo::EndPoint ep1, ep2, ep3;

        REQUIRE_EQ(true, ep1.parse("127.0.0.1:8080"));
        REQUIRE_EQ(true, ep2.parse("127.0.0.1:8080"));
        REQUIRE_EQ(true, ep3.parse("127.0.0.3:8080"));
        REQUIRE_EQ(ep1, ep2);
        REQUIRE_NE(ep1, ep3);

        REQUIRE_EQ(true, ep1.parse("unix:sock1.file"));
        REQUIRE_EQ(true, ep2.parse("unix:sock1.file"));
        REQUIRE_EQ(true, ep3.parse("unix:sock3.file"));
        REQUIRE_EQ(ep1, ep2);
        REQUIRE_NE(ep1, ep3);

        REQUIRE_EQ(true, ep1.parse("[::1]:2233"));
        REQUIRE_EQ(true, ep2.parse("[::1]:2233"));
        REQUIRE_EQ(true, ep3.parse("[::3]:2233"));
        REQUIRE_EQ(ep1, ep2);
        REQUIRE_NE(ep1, ep3);
    }

    TEST_CASE("EndPointTest, endpoint_sockaddr_conv_ipv4") {
        turbo::EndPoint ep;
        REQUIRE_EQ(true, ep.parse("1.2.3.4:8086"));

        in_addr expected_in_addr;
        bzero(&expected_in_addr, sizeof(expected_in_addr));
        expected_in_addr.s_addr = 0x04030201u;

        sockaddr_storage ss;
        sockaddr_in* in4 = (sockaddr_in*) &ss;

        memset(&ss, 'a', sizeof(ss));
        REQUIRE_EQ(true, ep.to_sockaddr(&ss));
        REQUIRE_EQ(AF_INET, ss.ss_family);
        REQUIRE_EQ(AF_INET, in4->sin_family);
        in_port_t port = htons(8086);
        REQUIRE_EQ(port, in4->sin_port);
        REQUIRE_EQ(0, memcmp(&in4->sin_addr, &expected_in_addr, sizeof(expected_in_addr)));

        sockaddr_storage ss2;
        socklen_t ss2_size = 0;
        memset(&ss2, 'b', sizeof(ss2));
        REQUIRE_EQ(true, ep.to_sockaddr(&ss2, &ss2_size));
        REQUIRE_EQ(ss2_size, sizeof(*in4));
        REQUIRE_EQ(0, memcmp(&ss2, &ss, sizeof(ss)));

        turbo::EndPoint ep2;
        REQUIRE_EQ(true, ep2.parse_sockaddr(&ss, sizeof(*in4)));
        REQUIRE_EQ(ep2, ep);

        REQUIRE_EQ(AF_INET,ep.get_endpoint_type());
    }

    TEST_CASE("EndPointTest, endpoint_sockaddr_conv_ipv6") {
        turbo::EndPoint ep;
        REQUIRE_EQ(true, ep.parse("[::1]:8086"));

        in6_addr expect_in6_addr;
        bzero(&expect_in6_addr, sizeof(expect_in6_addr));
        expect_in6_addr.s6_addr[15] = 1;

        sockaddr_storage ss;
        const sockaddr_in6* sa6 = (sockaddr_in6*) &ss;

        memset(&ss, 'a', sizeof(ss));
        REQUIRE_EQ(true, ep.to_sockaddr(&ss));
        REQUIRE_EQ(AF_INET6, ss.ss_family);
        REQUIRE_EQ(AF_INET6, sa6->sin6_family);
        in_port_t port = htons(8086);
        REQUIRE_EQ(port, sa6->sin6_port);
        REQUIRE_EQ(0u, sa6->sin6_flowinfo);
        REQUIRE_EQ(0, memcmp(&expect_in6_addr, &sa6->sin6_addr, sizeof(in6_addr)));
        REQUIRE_EQ(0u, sa6->sin6_scope_id);

        sockaddr_storage ss2;
        socklen_t ss2_size = 0;
        memset(&ss2, 'b', sizeof(ss2));
        REQUIRE_EQ(true, ep.to_sockaddr(&ss2, &ss2_size));
        REQUIRE_EQ(ss2_size, sizeof(*sa6));
        REQUIRE_EQ(0, memcmp(&ss2, &ss, sizeof(ss)));

        turbo::EndPoint ep2;
        REQUIRE_EQ(true, ep2.parse_sockaddr(&ss, sizeof(*sa6)));
        REQUIRE_EQ("[::1]:8086", ep2.to_string());

        REQUIRE_EQ(AF_INET6, ep.get_endpoint_type());
    }

    TEST_CASE("EndPointTest, endpoint_sockaddr_conv_unix") {
        turbo::EndPoint ep;
        REQUIRE_EQ(true, ep.parse("unix:sock.file"));

        sockaddr_storage ss;
        const sockaddr_un* un = (sockaddr_un*) &ss;

        memset(&ss, 'a', sizeof(ss));
        REQUIRE_EQ(true, ep.to_sockaddr( &ss));
        REQUIRE_EQ(AF_UNIX, ss.ss_family);
        REQUIRE_EQ(AF_UNIX, un->sun_family);
        REQUIRE_EQ(0, memcmp("sock.file", un->sun_path, 10));

        sockaddr_storage ss2;
        socklen_t ss2_size = 0;
        memset(&ss2, 'b', sizeof(ss2));
        REQUIRE_EQ(true, ep.to_sockaddr(&ss2, &ss2_size));
        REQUIRE_EQ(offsetof(struct sockaddr_un, sun_path) + strlen("sock.file") + 1, ss2_size);
        REQUIRE_EQ(0, memcmp(&ss2, &ss, sizeof(ss)));

        turbo::EndPoint ep2;
        REQUIRE_EQ(true, ep2.parse_sockaddr(&ss, sizeof(sa_family_t) + strlen(un->sun_path) + 1));
        REQUIRE_EQ("unix:sock.file", ep2.to_string());

        REQUIRE_EQ(AF_UNIX, ep.get_endpoint_type());
    }

    void concurrent_proc(void* p) {
        for (int i = 0; i < 10000; ++i) {
            turbo::EndPoint ep;
            std::string str("127.0.0.1:8080");
            REQUIRE_EQ(true, ep.parse(str.c_str()));
            REQUIRE_EQ(str, ep.to_string());

            str.assign("[::1]:8080");
            REQUIRE_EQ(true, ep.parse(str.c_str()));
            REQUIRE_EQ(str, ep.to_string());

            str.assign("unix:test.sock");
            REQUIRE_EQ(true, ep.parse(str.c_str()));
            REQUIRE_EQ(str, ep.to_string());
        }
        *(int*)p = 1;
    }

    TEST_CASE("EndPointTest, endpoint_concurrency") {
        const int T = 5;
        pthread_t tids[T];
        int rets[T] = {0};
        for (int i = 0; i < T; ++i) {
            pthread_create(&tids[i], nullptr, [](void* p) {
                concurrent_proc(p);
                return (void*)nullptr;
            }, &rets[i]);
        }
        for (int i = 0; i < T; ++i) {
            pthread_join(tids[i], nullptr);
            REQUIRE_EQ(1, rets[i]);
        }
    }

}  // namespace
