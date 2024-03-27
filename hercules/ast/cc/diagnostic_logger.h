// Copyright 2024 The titan-search Authors.
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

#include <collie/type_safe/reference.h>

#include <hercules/ast/cc/diagnostic.h>

namespace hercules::ccast
{
/// Base class for a [hercules::ccast::diagnostic]() logger.
///
/// Its task is controlling how diagnostic are being displayed.
class diagnostic_logger
{
public:
    /// \effects Creates it either as verbose or not.
    explicit diagnostic_logger(bool is_verbose = false) noexcept : verbose_(is_verbose) {}

    diagnostic_logger(const diagnostic_logger&)            = delete;
    diagnostic_logger& operator=(const diagnostic_logger&) = delete;
    virtual ~diagnostic_logger() noexcept                  = default;

    /// \effects Logs the diagnostic by invoking the `do_log()` member function.
    /// \returns Whether or not the diagnostic was logged.
    /// \notes `source` points to a string literal that gives additional context to what generates
    /// the message.
    bool log(const char* source, const diagnostic& d) const;

    /// \effects Sets whether or not the logger prints debugging diagnostics.
    void set_verbose(bool value) noexcept
    {
        verbose_ = value;
    }

    /// \returns Whether or not the logger prints debugging diagnostics.
    bool is_verbose() const noexcept
    {
        return verbose_;
    }

private:
    virtual bool do_log(const char* source, const diagnostic& d) const = 0;

    bool verbose_;
};

/// \returns The default logger object.
collie::ts::object_ref<const diagnostic_logger> default_logger() noexcept;

/// \returns The default verbose logger object.
collie::ts::object_ref<const diagnostic_logger> default_verbose_logger() noexcept;

/// A [hercules::ccast::diagnostic_logger]() that logs to `stderr`.
///
/// It prints all diagnostics in an implementation-defined format.
class stderr_diagnostic_logger final : public diagnostic_logger
{
public:
    using diagnostic_logger::diagnostic_logger;

private:
    bool do_log(const char* source, const diagnostic& d) const override;
};
} // namespace hercules::ccast
