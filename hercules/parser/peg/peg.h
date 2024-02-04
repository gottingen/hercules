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

#include <memory>
#include <string>
#include <vector>

#include "hercules/parser/ast.h"
#include "hercules/parser/cache.h"
#include "hercules/util/common.h"

namespace hercules::ast {

/// Parse a Seq code block with the appropriate file and position offsets.
StmtPtr parseCode(Cache *cache, const std::string &file, const std::string &code,
                  int line_offset = 0);
/// Parse a Seq code expression.
/// @return pair of ExprPtr and a string indicating format specification
/// (empty if not available).
std::pair<ExprPtr, std::string> parseExpr(Cache *cache, const std::string &code,
                                          const hercules::SrcInfo &offset);
/// Parse a Seq file.
StmtPtr parseFile(Cache *cache, const std::string &file);

/// Parse a OpenMP clause.
std::vector<CallExpr::Arg> parseOpenMP(Cache *cache, const std::string &code,
                                       const hercules::SrcInfo &loc);

} // namespace hercules::ast
