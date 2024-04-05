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
    #include <collie/log/details/tcp_client-windows.h>
#else
    #include <collie/log/details/tcp_client.h>
#endif

#include <chrono>
#include <functional>
#include <mutex>
#include <string>

#pragma once

// Simple tcp client sink
// Connects to remote address and send the formatted log.
// Will attempt to reconnect if connection drops.
// If more complicated behaviour is needed (i.e get responses), you can inherit it and override the
// sink_it_ method.

namespace clog {
namespace sinks {

struct tcp_sink_config {
    std::string server_host;
    int server_port;
    bool lazy_connect = false;  // if true connect on first log call instead of on construction

    tcp_sink_config(std::string host, int port)
        : server_host{std::move(host)},
          server_port{port} {}
};

template <typename Mutex>
class tcp_sink : public clog::sinks::base_sink<Mutex> {
public:
    // connect to tcp host/port or throw if failed
    // host can be hostname or ip address

    explicit tcp_sink(tcp_sink_config sink_config)
        : config_{std::move(sink_config)} {
        if (!config_.lazy_connect) {
            this->client_.connect(config_.server_host, config_.server_port);
        }
    }

    ~tcp_sink() override = default;

protected:
    void sink_it_(const clog::details::log_msg &msg) override {
        clog::memory_buf_t formatted;
        clog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        if (!client_.is_connected()) {
            client_.connect(config_.server_host, config_.server_port);
        }
        client_.send(formatted.data(), formatted.size());
    }

    void flush_() override {}
    tcp_sink_config config_;
    details::tcp_client client_;
};

using tcp_sink_mt = tcp_sink<std::mutex>;
using tcp_sink_st = tcp_sink<clog::details::null_mutex>;

}  // namespace sinks
}  // namespace clog
