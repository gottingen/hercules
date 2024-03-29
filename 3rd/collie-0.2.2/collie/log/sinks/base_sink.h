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
//
// base sink templated over a mutex (either dummy or real)
// concrete implementation should override the sink_it_() and flush_()  methods.
// locking is taken care of in this class - no locking needed by the
// implementers..
//

#include <collie/log/common.h>
#include <collie/log/details/log_msg.h>
#include <collie/log/sinks/sink.h>

namespace clog {
    namespace sinks {
        template<typename Mutex>
        class base_sink : public sink {
        public:
            base_sink();

            explicit base_sink(std::unique_ptr<clog::formatter> formatter);

            ~base_sink() override = default;

            base_sink(const base_sink &) = delete;

            base_sink(base_sink &&) = delete;

            base_sink &operator=(const base_sink &) = delete;

            base_sink &operator=(base_sink &&) = delete;

            void log(const details::log_msg &msg) final;

            void flush() final;

            void set_pattern(const std::string &pattern) final;

            void set_formatter(std::unique_ptr<clog::formatter> sink_formatter) final;

        protected:
            // sink formatter
            std::unique_ptr<clog::formatter> formatter_;
            Mutex mutex_;

            virtual void sink_it_(const details::log_msg &msg) = 0;

            virtual void flush_() = 0;

            virtual void set_pattern_(const std::string &pattern);

            virtual void set_formatter_(std::unique_ptr<clog::formatter> sink_formatter);
        };
    }  // namespace sinks
}  // namespace clog

#include <collie/log/sinks/base_sink-inl.h>
