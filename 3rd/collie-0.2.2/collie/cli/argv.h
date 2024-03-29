// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <vector>

#include <collie/cli/macros.h>

namespace collie::detail {
#ifdef _WIN32
    /// Decode and return UTF-8 argv from GetCommandLineW.
    inline std::vector<std::string> compute_win32_argv();
#endif
}  // namespace collie::detail

#include <collie/cli/impl/argv_inl.h>
