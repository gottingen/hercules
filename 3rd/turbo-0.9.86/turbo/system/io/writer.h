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
//
// Created by jeff on 24-1-8.
//

#ifndef TURBO_SYSTEM_IO_WRITER_H_
#define TURBO_SYSTEM_IO_WRITER_H_

#include <sys/uio.h>
#include "turbo/status/result_status.h"

namespace turbo {

    class IWriter {
    public:
        virtual ~IWriter() {}

        // Semantics of parameters and return value are same as writev(2) except that
        // there's no `fd'.
        // WriteV is required to submit data gathered by multiple appends in one
        // run and enable the possibility of atomic writes.
        virtual ResultStatus<ssize_t> writev(const iovec* iov, int iovcnt) = 0;
    };

}  // namespace turbo

#endif  // TURBO_SYSTEM_IO_WRITER_H_
