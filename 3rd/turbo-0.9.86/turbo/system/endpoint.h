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

#ifndef TURBO_SYSTEM_ENDPOINT_H_
#define TURBO_SYSTEM_ENDPOINT_H_

#include <netinet/in.h>                          // in_addr
#include <sys/un.h>                              // sockaddr_un
#include <iostream>                              // std::ostream
#include "turbo/status/status.h"
#include "turbo/status/result_status.h"
#include "turbo/format/format.h"

namespace turbo {
    namespace system_internal {
        class ExtendedEndPoint;
    }

    struct ListenOption {
        bool reuse_addr{true};
        bool reuse_port{false};
        bool reuse_uds{false};
    };
    // Type of an IP address
    typedef struct in_addr ip_t;

    static constexpr int MAX_DOMAIN_LENGTH = 253;

    struct IPStr {
        const char *c_str() const { return _buf; }

        char _buf[INET_ADDRSTRLEN];
    };

    class EndPoint;

    class IPAddr {
    public:

        static constexpr ip_t IP_ANY = {INADDR_ANY};
        static constexpr ip_t IP_NONE = {INADDR_NONE};

        constexpr IPAddr() = default;

        constexpr IPAddr(ip_t ip) : _ip(ip) {}

        constexpr IPAddr(uint32_t ip) : _ip(ip_t{ip}) {}

        // Convert string `ip_str' to ip_t *ip.
        // `ip_str' is in IPv4 dotted-quad format: `127.0.0.1', `10.23.249.73' ...
        // Returns false on success, true otherwise.
        bool parse(const char *ip_str);

        // Convert `hostname' to ip_t *ip. If `hostname' is NULL, use hostname
        // of this machine.
        // `hostname' is typically in this form: `tc-cm-et21.tc' `db-cos-dev.db01' ...
        // Returns false on success, true otherwise.
        bool parse_hostname(const char *hostname);

        constexpr uint32_t num() const { return _ip.s_addr; }

        constexpr ip_t ip_struct() const { return _ip; }

        constexpr bool is_any() const { return _ip.s_addr == IP_ANY.s_addr; }

        constexpr bool is_none() const { return _ip.s_addr == IP_NONE.s_addr; }

        bool to_hostname(char *hostname, size_t hostname_len) const;

        bool to_hostname(std::string &hostname) const;

        IPStr to_str() const;

        std::string to_string() const;

    public:
        static IPAddr my_ip();

        static const char *my_ip_cstr();

        static const char *my_hostname();

        constexpr static IPAddr any() { return IPAddr(IP_ANY); }

        constexpr static IPAddr none() { return IPAddr(IP_NONE); }

        friend class EndPoint;

    private:
        ip_t _ip{IP_ANY};
    };

    inline std::string IPAddr::to_string() const {
        return std::string(to_str().c_str());
    }

    struct EndPointStr {
        const char *c_str() const { return _buf; }

        char _buf[sizeof("unix:") + sizeof(sockaddr_un::sun_path)];
    };

    // For IPv4 endpoint, ip and port are real things.
    // For UDS/IPv6 endpoint, to keep ABI compatibility, ip is ResourceId, and port is a special flag.
    // See str2endpoint implementation for details.

    class EndPoint {
    public:
        constexpr EndPoint() = default;

        EndPoint(IPAddr ip2, int port2);

        explicit EndPoint(int port) : EndPoint(IPAddr::any(), port) {}

        explicit EndPoint(const sockaddr_in &in)
                : _ip(in.sin_addr), _port(ntohs(in.sin_port)) {}

        EndPoint(const EndPoint &);

        ~EndPoint();

        void operator=(const EndPoint &);

        void reset(void);

        // Convert EndPoint to c-style string. Notice that you can serialize
        // EndPoint to std::ostream directly. Use this function when you don't
        // have streaming log.
        // Example: printf("point=%s\n", endpoint2str(point).c_str());
        EndPointStr to_str() const;

        std::string to_string() const;

        constexpr IPAddr ip() const { return _ip; }

        IPStr ip_str() const { return _ip.to_str(); }

        std::string ip_string() const { return _ip.to_string(); }

        constexpr int port() const { return _port; }

        bool to_hostname(char *hostname, size_t hostname_len) const;

        bool to_hostname(std::string &hostname) const;

        bool to_sockaddr(struct sockaddr_storage *ss, socklen_t *size = nullptr) const;

        // Get EndPoint type (AF_INET/AF_INET6/AF_UNIX)
        sa_family_t get_endpoint_type() const;

        // Check if endpoint is extended.
        bool is_endpoint_extended() const;

        // Create a TCP socket and connect it to `server'. Write port of this side
        // into `self_port' if it's not NULL.
        // Returns the socket descriptor, -1 otherwise and errno is set.
        turbo::ResultStatus<int> tcp_connect(int *self_port);

        turbo::ResultStatus<int> tcp_listen(ListenOption option = ListenOption());

    public:
        ///////////////////////////// parser ///////////////////////////////////////
        bool parse(const char *ip_and_port_str);

        bool parse(const char *ip_str, int port);

        bool parse_hostname(const char *ip_and_port_str);

        bool parse_hostname(const char *ip_str, int port);

        bool parse_sockaddr(struct sockaddr_storage *ss, socklen_t size);

        bool parse_ip(const char *ip_str);

        void set(IPAddr ip, int port);

        void set(int port);

        // Get the local end of a socket connection
        bool local_side(int fd);

        // Get the other end of a socket connection
        bool remote_side(int fd);

    public:
        // Get the local end of a socket connection
        static bool test_local_side(int fd);

        // Get the other end of a socket connection
        static bool test_remote_side(int fd);
        IPAddr _ip;
        int _port{0};
    };
    static_assert(sizeof(EndPoint) == sizeof(IPAddr)+ sizeof(int),
                  "EndPoint size mismatch with the one in POD-style, may cause ABI problem");

    inline std::string EndPoint::to_string() const {
        return std::string(to_str().c_str());
    }
    template<typename H>
    inline H hash_value(H h, const IPAddr &ip) {
        return H::combine(std::move(h), ip.num());
    }

    template<typename H>
    inline H hash_value(H h, const EndPoint &point) {
        return H::combine(std::move(h), point.ip(), point.port());
    }


    inline bool operator<(turbo::IPAddr lhs, turbo::IPAddr rhs) {
        return lhs.num() < rhs.num();
    }

    inline bool operator>(turbo::IPAddr lhs, turbo::IPAddr rhs) {
        return rhs < lhs;
    }

    inline bool operator>=(turbo::IPAddr lhs, turbo::IPAddr rhs) {
        return !(lhs < rhs);
    }

    inline bool operator<=(turbo::IPAddr lhs, turbo::IPAddr rhs) {
        return !(rhs < lhs);
    }

    inline bool operator==(turbo::IPAddr lhs, turbo::IPAddr rhs) {
        return lhs.num() == rhs.num();
    }

    inline bool operator!=(turbo::IPAddr lhs, turbo::IPAddr rhs) {
        return !(lhs == rhs);
    }

    inline std::ostream &operator<<(std::ostream &os, const turbo::IPStr &ip_str) {
        return os << ip_str.c_str();
    }

    inline std::ostream &operator<<(std::ostream &os, turbo::IPAddr ip) {
        return os << ip.num();
    }

    inline bool operator<(EndPoint p1, EndPoint p2) {
        return (p1.ip() != p2.ip()) ? (p1.ip() < p2.ip()) : (p1.port() < p2.port());
    }

    inline bool operator>(EndPoint p1, EndPoint p2) {
        return p2 < p1;
    }

    inline bool operator<=(EndPoint p1, EndPoint p2) {
        return !(p2 < p1);
    }

    inline bool operator>=(EndPoint p1, EndPoint p2) {
        return !(p1 < p2);
    }

    inline bool operator==(EndPoint p1, EndPoint p2) {
        return p1.ip() == p2.ip() && p1.port() == p2.port();
    }

    inline bool operator!=(EndPoint p1, EndPoint p2) {
        return !(p1 == p2);
    }

    inline std::ostream &operator<<(std::ostream &os, const EndPoint &ep) {
        return os << ep.to_str().c_str();
    }

    inline std::ostream &operator<<(std::ostream &os, const EndPointStr &ep_str) {
        return os << ep_str.c_str();
    }

    template<>
    struct formatter<IPAddr> : formatter<std::string> {
        template<typename FormatContext>
        auto format(const IPAddr &p, FormatContext &ctx) {
            return formatter<std::string>::format(p.to_str().c_str(), ctx);
        }
    };

    template<>
    struct formatter<EndPoint> : formatter<std::string> {
        template<typename FormatContext>
        auto format(const EndPoint &p, FormatContext &ctx) {
            return formatter<std::string>::format(p.to_str().c_str(), ctx);
        }
    };

    // turbo_parse_flag()
    //
    // Parses a command-line flag string representation `text` into a Duration
    // value. EndPoint flags must be specified in a format that is valid input for
    // `turbo::ParseDuration()`.
    bool turbo_parse_flag(std::string_view text, EndPoint *dst, std::string *error);


    // turbo_unparse_flag()
    //
    // Unparses a EndPoint value into a command-line string representation using
    // the format specified by `turbo::ParseDuration()`.
    std::string turbo_unparse_flag(EndPoint d);

    // turbo_parse_flag()
    //
    // Parses a command-line flag string representation `text` into a Duration
    // value. IPAddr flags must be specified in a format that is valid input for
    // `turbo::ParseDuration()`.
    bool turbo_parse_flag(std::string_view text, IPAddr *dst, std::string *error);


    // turbo_unparse_flag()
    //
    // Unparses a IPAddr value into a command-line string representation using
    // the format specified by `turbo::ParseDuration()`.
    std::string turbo_unparse_flag(IPAddr d);
}  // namespace turbo

#endif  // TURBO_SYSTEM_ENDPOINT_H_
