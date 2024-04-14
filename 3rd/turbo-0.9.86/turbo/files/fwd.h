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

#ifndef TURBO_FILES_FWD_H_
#define TURBO_FILES_FWD_H_

#include "turbo/files/internal/filesystem.h"
#include "turbo/status/result_status.h"
#include "turbo/files/file_option.h"
#include "turbo/files/file_event_listener.h"
#include "turbo/files/internal/file_reader.h"
#include "turbo/files/internal/file_writer.h"
#include "turbo/files/internal/fwd.h"
#include <cstddef>

namespace turbo {

    template<typename Tag>
    struct FileAdapter;
}  // namespace turbo

#endif  // TURBO_FILES_FWD_H_
