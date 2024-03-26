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

#include <algorithm>
#include <cstdio>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <gc.h>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "hercules/hir/analyze/dataflow/capture.h"
#include "hercules/hir/analyze/dataflow/reaching.h"
#include "hercules/hir/util/inlining.h"
#include "hercules/hir/util/irtools.h"
#include "hercules/hir/util/operator.h"
#include "hercules/hir/util/outlining.h"
#include "hercules/compiler/compiler.h"
#include "hercules/compiler/error.h"
#include "hercules/parser/common.h"
#include "hercules/util/common.h"

#include "gtest/gtest.h"

// clang-format on
std::string argv0;

int main(int argc, char *argv[]) {
    argv0 = hercules::ast::executable_path(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
