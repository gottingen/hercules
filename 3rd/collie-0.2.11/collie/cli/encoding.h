// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <collie/cli/macros.h>
#include <collie/filesystem/fs.h>
#include <string>
#include <string_view>

namespace collie {

    /// Convert a wide string to a narrow string.
    inline std::string narrow(const std::wstring &str);
    inline std::string narrow(const wchar_t *str);
    inline std::string narrow(const wchar_t *str, std::size_t size);

    /// Convert a narrow string to a wide string.
    inline std::wstring widen(const std::string &str);
    inline std::wstring widen(const char *str);
    inline std::wstring widen(const char *str, std::size_t size);

    inline std::string narrow(std::wstring_view str);
    inline std::wstring widen(std::string_view str);

    /// Convert a char-string to a native path correctly.
    inline collie::filesystem::path to_path(std::string_view str);

}  // namespace collie

#include <collie/cli/impl/encoding_inl.h>
