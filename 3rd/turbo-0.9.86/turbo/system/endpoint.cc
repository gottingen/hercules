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

#include "turbo/system/endpoint.h"
#include "turbo/system/io.h"
#include "turbo/system/internal/endpoint_internal.h"
#include "turbo/strings/fixed_string.h"

namespace turbo {

    turbo::Status TURBO_WEAK fiber_connect(
            int sockfd, const struct sockaddr *serv_addr, socklen_t addrlen) {
        return connect(sockfd, serv_addr, addrlen) == 0 ? turbo::ok_status() : turbo::errno_to_status(errno, "");
    }

    using turbo::system_internal::ExtendedEndPoint;

    void EndPoint::set(IPAddr ip, int port) {
        _ip = ip;
        _port = port;
        if (ExtendedEndPoint::is_extended(*this)) {
            ExtendedEndPoint *eep = ExtendedEndPoint::address(*this);
            if (eep) {
                eep->inc_ref();
            } else {
                _ip = IPAddr::any();
                _port = 0;
            }
        }
    }

    void EndPoint::set(int port) {
        _port = port;
    }

    void EndPoint::reset(void) {
        if (ExtendedEndPoint::is_extended(*this)) {
            ExtendedEndPoint *eep = ExtendedEndPoint::address(*this);
            if (eep) {
                eep->dec_ref();
            }
        }
        _ip = IPAddr::any();
        _port = 0;
    }

    EndPoint::EndPoint(IPAddr ip2, int port2) : _ip(ip2), _port(port2) {
        // Should never construct an extended endpoint by this way
        if (ExtendedEndPoint::is_extended(*this)) {
            TLOG_CHECK(0, "EndPoint construct with value that points to an extended EndPoint");
            _ip = IPAddr::any();
            _port = 0;
        }
    }

    EndPoint::EndPoint(const EndPoint &rhs) {
        set(rhs._ip, rhs._port);
    }

    EndPoint::~EndPoint() {
        reset();
    }

    void EndPoint::operator=(const EndPoint &rhs) {
        reset();
        set(rhs._ip, rhs._port);
    }

    bool IPAddr::parse(const char *ip_str) {
        if (ip_str != nullptr) {
            for (; isspace(*ip_str); ++ip_str);
            int rc = inet_pton(AF_INET, ip_str, &_ip);
            if (rc > 0) {
                return true;
            }
        }
        return false;
    }

    IPStr IPAddr::to_str() const {
        IPStr str;
        if (inet_ntop(AF_INET, &_ip, str._buf, INET_ADDRSTRLEN) == nullptr) {
            return any().to_str();
        }
        return str;
    }

    bool IPAddr::to_hostname(char *host, size_t host_len) const {
        if (host == nullptr || host_len == 0) {
            errno = EINVAL;
            return -1;
        }
        sockaddr_in sa;
        bzero((char *) &sa, sizeof(sa));
        sa.sin_family = AF_INET;
        sa.sin_port = 0;    // useless since we don't need server_name
        sa.sin_addr = _ip;
        if (getnameinfo((const sockaddr *) &sa, sizeof(sa),
                        host, host_len, nullptr, 0, NI_NAMEREQD) != 0) {
            return -1;
        }
        return 0;
    }

    bool IPAddr::to_hostname(std::string &host) const {
        char buf[128];
        if (TURBO_LIKELY(to_hostname(buf, sizeof(buf)))) {
            host.assign(buf);
            return true;
        }
        return false;
    }

    EndPointStr EndPoint::to_str() const {
        EndPointStr str;
        if (ExtendedEndPoint::is_extended(*this)) {
            ExtendedEndPoint *eep = ExtendedEndPoint::address(*this);
            if (eep) {
                eep->to(&str);
            } else {
                str._buf[0] = '\0';
            }
            return str;
        }
        if (inet_ntop(AF_INET, &_ip._ip, str._buf, INET_ADDRSTRLEN) == nullptr) {
            return EndPoint(IPAddr::none(), 0).to_str();
        }
        char *buf = str._buf + strlen(str._buf);
        *buf++ = ':';
        snprintf(buf, 16, "%d", _port);
        return str;
    }

    bool IPAddr::parse_hostname(const char *hostname) {
        char buf[256];
        if (nullptr == hostname) {
            if (gethostname(buf, sizeof(buf)) < 0) {
                return false;
            }
            hostname = buf;
        } else {
            // skip heading space
            for (; isspace(*hostname); ++hostname);
        }

#if defined(TURBO_PLATFORM_OSX)
        // gethostbyname on MAC is thread-safe (with current usage) since the
        // returned hostent is TLS. Check following link for the ref:
        // https://lists.apple.com/archives/darwin-dev/2006/May/msg00008.html
        struct hostent* result = gethostbyname(hostname);
        if (result == nullptr) {
            return false;
        }
#else
        int aux_buf_len = 1024;
        std::unique_ptr<char[]> aux_buf(new char[aux_buf_len]);
        int ret = 0;
        int error = 0;
        struct hostent ent;
        struct hostent *result = nullptr;
        do {
            result = nullptr;
            error = 0;
            ret = gethostbyname_r(hostname,
                                  &ent,
                                  aux_buf.get(),
                                  aux_buf_len,
                                  &result,
                                  &error);
            if (ret != ERANGE) { // aux_buf is not long enough
                break;
            }
            aux_buf_len *= 2;
            aux_buf.reset(new char[aux_buf_len]);
        } while (1);
        if (ret != 0 || result == nullptr) {
            return false;
        }
#endif // defined(TURBO_PLATFORM_OSX)
        // Only fetch the first address here
        bcopy((char *) result->h_addr, (char *) &_ip, result->h_length);
        return true;
    }

    struct MyAddressInfo {
        char my_hostname[256];
        IPAddr my_ip;
        IPStr my_ip_str;

        MyAddressInfo() {
            my_ip = IPAddr::any();
            if (gethostname(my_hostname, sizeof(my_hostname)) < 0) {
                my_hostname[0] = '\0';
            } else if (!my_ip.parse_hostname(my_hostname)) {
                my_ip = IPAddr::any();;
            }
            my_ip_str = my_ip.to_str();
        }

        static MyAddressInfo *get_instance() {
            static MyAddressInfo my_addr;
            return &my_addr;
        }

    };


    IPAddr IPAddr::my_ip() {
        return MyAddressInfo::get_instance()->my_ip;
    }

    const char *IPAddr::my_ip_cstr() {
        return MyAddressInfo::get_instance()->my_ip_str.c_str();
    }

    const char *IPAddr::my_hostname() {
        return MyAddressInfo::get_instance()->my_hostname;
    }

    bool EndPoint::parse(const char *str) {
        if (ExtendedEndPoint::create(str, this)) {
            return true;
        }

        // Should be enough to hold ip address
        char buf[64];
        size_t i = 0;
        for (; i < sizeof(buf) && str[i] != '\0' && str[i] != ':'; ++i) {
            buf[i] = str[i];
        }
        if (i >= sizeof(buf) || str[i] != ':') {
            return false;
        }
        buf[i] = '\0';
        if (TURBO_UNLIKELY(!_ip.parse(buf))) {
            return false;
        }
        ++i;
        char *end = nullptr;
        _port = strtol(str + i, &end, 10);
        if (end == str + i) {
            return false;
        } else if (*end) {
            for (++end; isspace(*end); ++end);
            if (*end) {
                return false;
            }
        }
        if (_port < 0 || _port > 65535) {
            return false;
        }
        return true;
    }

    bool EndPoint::parse(const char *ip_str, int port) {
        if (ExtendedEndPoint::create(ip_str, port, this)) {
            return true;
        }

        if (TURBO_UNLIKELY(!_ip.parse(ip_str))) {
            return false;
        }
        if (port < 0 || port > 65535) {
            return false;
        }
        _port = port;
        return true;
    }

    bool EndPoint::parse_hostname(const char *str) {
        // Should be enough to hold ip address
        // The definitive descriptions of the rules for forming domain names appear in RFC 1035, RFC 1123, RFC 2181,
        // and RFC 5892. The full domain name may not exceed the length of 253 characters in its textual representation
        // (Domain Names - Domain Concepts and Facilities. IETF. doi:10.17487/RFC1034. RFC 1034.).
        // For cacheline optimize, use buf size as 256;
        char buf[256];
        size_t i = 0;
        for (; i < MAX_DOMAIN_LENGTH && str[i] != '\0' && str[i] != ':'; ++i) {
            buf[i] = str[i];
        }

        if (TURBO_UNLIKELY(i >= MAX_DOMAIN_LENGTH || str[i] != ':')) {
            return false;
        }

        buf[i] = '\0';
        if (TURBO_UNLIKELY(!_ip.parse_hostname(buf))) {
            return false;
        }
        if (str[i] == ':') {
            ++i;
        }
        char *end = nullptr;
        _port = strtol(str + i, &end, 10);
        if (end == str + i) {
            return false;
        } else if (*end) {
            for (; isspace(*end); ++end);
            if (TURBO_UNLIKELY(*end)) {
                return false;
            }
        }
        if (TURBO_UNLIKELY(_port < 0 || _port > 65535)) {
            return false;
        }
        return true;
    }

    bool EndPoint::parse_hostname(const char *name_str, int port) {
        if (TURBO_UNLIKELY(!_ip.parse_hostname(name_str))) {
            return false;
        }
        if (TURBO_UNLIKELY(port < 0 || port > 65535)) {
            return false;
        }
        _port = port;
        return true;
    }

    bool EndPoint::parse_sockaddr(struct sockaddr_storage *ss, socklen_t size) {
        if (ss->ss_family == AF_INET) {
            *this = EndPoint(*(sockaddr_in *) ss);
            return true;
        }
        if (ExtendedEndPoint::create(ss, size, this)) {
            return true;
        }
        return false;
    }

    bool EndPoint::parse_ip(const char *ip_str) {
        return _ip.parse(ip_str);
    }

    bool EndPoint::to_hostname(char *host, size_t host_len) const {
        if (ExtendedEndPoint::is_extended(*this)) {
            ExtendedEndPoint *eep = ExtendedEndPoint::address(*this);
            if (eep) {
                return eep->to_hostname(host, host_len);
            }
            return false;
        }

        if (TURBO_LIKELY(_ip.to_hostname(host, host_len))) {
            size_t len = strlen(host);
            if (len + 1 < host_len) {
                ::snprintf(host + len, host_len - len, ":%d", _port);
            }
            return true;
        }
        return false;
    }

    bool EndPoint::to_hostname(std::string &host) const {
        char buf[256];
        if (to_hostname(buf, sizeof(buf)) == 0) {
            host.assign(buf);
            return true;
        }
        return false;
    }

    bool EndPoint::to_sockaddr(struct sockaddr_storage *ss, socklen_t *size) const {
        bzero(ss, sizeof(*ss));
        if (ExtendedEndPoint::is_extended(*this)) {
            ExtendedEndPoint *eep = ExtendedEndPoint::address(*this);
            if (!eep) {
                return false;
            }
            int ret = eep->to(ss);
            if (ret < 0) {
                return false;
            }
            if (size) {
                *size = static_cast<socklen_t>(ret);
            }
            return true;
        }
        struct sockaddr_in *in4 = (struct sockaddr_in *) ss;
        in4->sin_family = AF_INET;
        in4->sin_addr = _ip._ip;
        in4->sin_port = htons(_port);
        if (size) {
            *size = sizeof(*in4);
        }
        return true;
    }

    sa_family_t EndPoint::get_endpoint_type() const {
        if (ExtendedEndPoint::is_extended(*this)) {
            ExtendedEndPoint *eep = ExtendedEndPoint::address(*this);
            if (eep) {
                return eep->family();
            }
            return AF_UNSPEC;
        }
        return AF_INET;
    }

    bool EndPoint::is_endpoint_extended() const {
        return ExtendedEndPoint::is_extended(*this);
    }


    turbo::ResultStatus<int> EndPoint::tcp_connect(int *self_port) {
        struct sockaddr_storage serv_addr;
        socklen_t serv_addr_size = 0;
        if (TURBO_UNLIKELY(!to_sockaddr(&serv_addr, &serv_addr_size))) {
            return turbo::make_status(kEINVAL,"Fail to convert EndPoint to sockaddr");
        }
        FDGuard sockfd(socket(serv_addr.ss_family, SOCK_STREAM, 0));
        if (sockfd < 0) {
            return turbo::make_status();
        }
        int rc = 0;
        if (fiber_connect != nullptr) {
            rc = fiber_connect(sockfd, (struct sockaddr *) &serv_addr, serv_addr_size).ok() ? 0 : -1;
        } else {
            rc = ::connect(sockfd, (struct sockaddr *) &serv_addr, serv_addr_size);
        }
        if (rc < 0) {
            turbo::make_status();
        }
        if (self_port != nullptr) {
            EndPoint pt;
            if (TURBO_LIKELY(pt.local_side(sockfd))) {
                *self_port = pt._port;
            } else {
                TLOG_CHECK(false, "Fail to get the local port of sockfd={}", (int) sockfd);
            }
        }
        return sockfd.release();
    }

    turbo::ResultStatus<int> EndPoint::tcp_listen(ListenOption option) {
        struct sockaddr_storage serv_addr;
        socklen_t serv_addr_size = 0;
        if (TURBO_UNLIKELY(!to_sockaddr(&serv_addr, &serv_addr_size))) {
            return turbo::make_status(kEINVAL,"Fail to convert EndPoint to sockaddr");
        }
        FDGuard sockfd(::socket(serv_addr.ss_family, SOCK_STREAM, 0));
        if (sockfd < 0) {
            return turbo::make_status();
        }

        if (option.reuse_addr) {
#if defined(SO_REUSEADDR)
            const int on = 1;
            if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR,
                           &on, sizeof(on)) != 0) {
                return turbo::make_status();
            }
#else
            TLOG_ERROR("Missing def of SO_REUSEADDR while -reuse_addr is on");
            return turbo::make_status();
#endif
        }

        if (option.reuse_port) {
#if defined(SO_REUSEPORT)
            const int on = 1;
            if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT,
                           &on, sizeof(on)) != 0) {
                TLOG_WARN("Fail to setsockopt SO_REUSEPORT of sockfd={}", (int) sockfd);
            }
#else
            TLOG_ERROR("Missing def of SO_REUSEPORT while -reuse_port is on");
            return turbo::make_status();
#endif
        }

        if (option.reuse_uds && serv_addr.ss_family == AF_UNIX) {
            ::unlink(((sockaddr_un *) &serv_addr)->sun_path);
        }

        if (::bind(sockfd, (struct sockaddr *) &serv_addr, serv_addr_size) != 0) {
            return turbo::make_status();
        }
        if (::listen(sockfd, 65535) != 0) {
            //             ^^^ kernel would silently truncate backlog to the value
            //             defined in /proc/sys/net/core/somaxconn if it is less
            //             than 65535
            return turbo::make_status();
        }
        return sockfd.release();
    }

    bool EndPoint::local_side(int fd) {
        struct sockaddr_storage addr;
        socklen_t socklen = sizeof(addr);
        const int rc = getsockname(fd, (struct sockaddr *) &addr, &socklen);
        if (rc != 0) {
            return false;
        }
        return parse_sockaddr(&addr, socklen);
    }

    bool EndPoint::remote_side(int fd) {
        struct sockaddr_storage addr;
        bzero(&addr, sizeof(addr));
        socklen_t socklen = sizeof(addr);
        const int rc = getpeername(fd, (struct sockaddr *) &addr, &socklen);
        if (rc != 0) {
            return false;
        }
        return parse_sockaddr(&addr, socklen);
    }

    bool EndPoint::test_local_side(int fd) {
        struct sockaddr_storage addr;
        socklen_t socklen = sizeof(addr);
        const int rc = getsockname(fd, (struct sockaddr *) &addr, &socklen);
        if (rc != 0) {
            return false;
        }
        return true;
    }

    bool EndPoint::test_remote_side(int fd) {
        struct sockaddr_storage addr;
        bzero(&addr, sizeof(addr));
        socklen_t socklen = sizeof(addr);
        const int rc = getpeername(fd, (struct sockaddr *) &addr, &socklen);
        if (rc != 0) {
            return false;
        }
        return true;
    }


    // turbo_parse_flag()
    //
    // Parses a command-line flag string representation `text` into a Duration
    // value. EndPoint flags must be specified in a format that is valid input for
    // `turbo::ParseDuration()`.
    bool turbo_parse_flag(std::string_view text, EndPoint *dst, std::string *error) {
        std::string str(text);
        // avoid parse error when text is not null-terminated
        return dst->parse(str.c_str());
    }


    // turbo_unparse_flag()
    //
    // Unparses a EndPoint value into a command-line string representation using
    // the format specified by `turbo::ParseDuration()`.
    std::string turbo_unparse_flag(EndPoint d) {
        return d.to_str().c_str();
    }

    // turbo_parse_flag()
    //
    // Parses a command-line flag string representation `text` into a Duration
    // value. IPAddr flags must be specified in a format that is valid input for
    // `turbo::ParseDuration()`.
    bool turbo_parse_flag(std::string_view text, IPAddr *dst, std::string *error) {
        std::string str(text);
        // avoid parse error when text is not null-terminated
        return dst->parse(str.c_str());
    }


    // turbo_unparse_flag()
    //
    // Unparses a IPAddr value into a command-line string representation using
    // the format specified by `turbo::ParseDuration()`.
    std::string turbo_unparse_flag(IPAddr d) {
        return d.to_str().c_str();
    }
}  // namespace turbo

