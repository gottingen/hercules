/****************************************************************
 * Copyright (c) 2023, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#ifndef HERCULES_ENGINE_CORE_ERROR_CODE_H_
#define HERCULES_ENGINE_CORE_ERROR_CODE_H_

#include <string>
#include <melon/base/result_status.h>
#include "hercules/engine/proto/error_code.pb.h"

namespace hercules::engine {

    std::string error_string(hercules::proto::ErrorCode ec);

    inline melon::result_status error_status(hercules::proto::ErrorCode ec) {
        return melon::result_status(ec, error_string(ec));
    }
}  //namespace hercules::engine

#endif  // HERCULES_ENGINE_CORE_ERROR_CODE_H_
