/****************************************************************
 * Copyright (c) 2023, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "hercules/engine/core/error_code.h"

namespace hercules::engine {


    static const std::vector<std::string> g_error_array{
            " update doc fail",
            "do not support updata and insert",
            "doc invalid",
            "overflow capacity",
            "index initialize fail",
            "unknown error"
    };

    std::string error_string(hercules::proto::ErrorCode ec) {
        if (ec == hercules::proto::EC_OK) {
            return "ok";
        }
        int offset = static_cast<int>(ec) - 800;
        if (offset < 0 || static_cast<size_t>(offset) >= g_error_array.size()) {
            return "bad error code";
        }
        return g_error_array[offset];
    }
}  // hercules::engine



