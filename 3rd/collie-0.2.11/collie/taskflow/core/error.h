// Copyright 2024 The Elastic AI Search Authors.
//
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

#include <iostream>
#include <sstream>
#include <exception>

#include <collie/taskflow/utility/stream.h>

namespace collie::tf {

// Procedure: throw_se
// Throws the system error under a given error code.
template <typename... ArgsT>
//void throw_se(const char* fname, const size_t line, Error::Code c, ArgsT&&... args) {
void throw_re(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss;
  oss << "[" << fname << ":" << line << "] ";
  //ostreamize(oss, std::forward<ArgsT>(args)...);
  (oss << ... << args);
  throw std::runtime_error(oss.str());
}

}  // ------------------------------------------------------------------------

#define TF_THROW(...) collie::tf::throw_re(__FILE__, __LINE__, __VA_ARGS__);

