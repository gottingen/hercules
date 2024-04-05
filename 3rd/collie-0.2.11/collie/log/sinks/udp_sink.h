// Copyright 2024 The Elastic-AI Authors.
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
#pragma once

#include <collie/log/common.h>
#include <collie/log/details/null_mutex.h>
#include <collie/log/sinks/base_sink.h>
#ifdef _WIN32
    #include <collie/log/details/udp_client-windows.h>
#else
    #include <collie/log/details/udp_client.h>
#endif

#include <chrono>
#include <functional>
#include <mutex>
#include <string>

// Simple udp client sink
// Sends formatted log via udp

namespace clog {
namespace sinks {

struct udp_sink_config {
    std::string server_host;
    uint16_t server_port;

    udp_sink_config(std::string host, uint16_t port)
        : server_host{std::move(host)},
          server_port{port} {}
};

template <typename Mutex>
class udp_sink : public clog::sinks::base_sink<Mutex> {
public:
    // host can be hostname or ip address
    explicit udp_sink(udp_sink_config sink_config)
        : client_{sink_config.server_host, sink_config.server_port} {}

    ~udp_sink() override = default;

protected:
    void sink_it_(const clog::details::log_msg &msg) override {
        clog::memory_buf_t formatted;
        clog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        client_.send(formatted.data(), formatted.size());
    }

    void flush_() override {}
    details::udp_client client_;
};

using udp_sink_mt = udp_sink<std::mutex>;
using udp_sink_st = udp_sink<clog::details::null_mutex>;

}  // namespace sinks

//
// factory functions
//
template <typename Factory = clog::synchronous_factory>
inline std::shared_ptr<logger> udp_logger_mt(const std::string &logger_name,
                                             sinks::udp_sink_config skin_config) {
    return Factory::template create<sinks::udp_sink_mt>(logger_name, skin_config);
}

}  // namespace clog
