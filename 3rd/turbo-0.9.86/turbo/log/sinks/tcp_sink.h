// Copyright 2023 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#include "turbo/log/common.h"
#include <turbo/log/sinks/base_sink.h>
#include "turbo/log/details/null_mutex.h"
#ifdef _WIN32
#    include <turbo/log/details/tcp_client-windows.h>
#else
#    include <turbo/log/details/tcp_client.h>
#endif

#include <mutex>
#include <string>
#include <chrono>
#include <functional>

#pragma once

// Simple tcp client sink
// Connects to remote address and send the formatted log.
// Will attempt to reconnect if connection drops.
// If more complicated behaviour is needed (i.e get responses), you can inherit it and override the sink_it_ method.

namespace turbo::tlog {
namespace sinks {

struct tcp_sink_config
{
    std::string server_host;
    int server_port;
    bool lazy_connect = false; // if true connect on first log call instead of on construction

    tcp_sink_config(std::string host, int port)
        : server_host{std::move(host)}
        , server_port{port}
    {}
};

template<typename Mutex>
class tcp_sink : public turbo::tlog::sinks::base_sink<Mutex>
{
public:
    // connect to tcp host/port or throw if failed
    // host can be hostname or ip address

    explicit tcp_sink(tcp_sink_config sink_config)
        : config_{std::move(sink_config)}
    {
        if (!config_.lazy_connect)
        {
            this->client_.connect(config_.server_host, config_.server_port);
        }
    }

    ~tcp_sink() override = default;

protected:
    void sink_it_(const turbo::tlog::details::log_msg &msg) override
    {
        turbo::tlog::memory_buf_t formatted;
        turbo::tlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        if (!client_.is_connected())
        {
            client_.connect(config_.server_host, config_.server_port);
        }
        client_.send(formatted.data(), formatted.size());
    }

    void flush_() override {}
    tcp_sink_config config_;
    details::tcp_client client_;
};

using tcp_sink_mt = tcp_sink<std::mutex>;
using tcp_sink_st = tcp_sink<turbo::tlog::details::null_mutex>;

} // namespace sinks
} // namespace turbo::tlog
