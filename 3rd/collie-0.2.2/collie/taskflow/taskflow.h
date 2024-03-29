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

#include <collie/taskflow/core/executor.h>
#include <collie/taskflow/core/async.h>
#include <collie/taskflow/algorithm/critical.h>

/**
@dir taskflow
@brief root taskflow include dir
*/

/**
@dir taskflow/core
@brief taskflow core include dir
*/

/**
@dir taskflow/algorithm
@brief taskflow algorithms include dir
*/

/**
@dir taskflow/cuda
@brief taskflow CUDA include dir
*/

/**
@file taskflow/taskflow.hpp
@brief main taskflow include file
*/

// TF_VERSION % 100 is the patch level
// TF_VERSION / 100 % 1000 is the minor version
// TF_VERSION / 100000 is the major version

// current version: 3.7.0
#define TF_VERSION 300700

#define TF_MAJOR_VERSION TF_VERSION/100000
#define TF_MINOR_VERSION TF_VERSION/100%1000
#define TF_PATCH_VERSION TF_VERSION%100

/**
@brief taskflow namespace
*/
namespace collie::tf {

/**
@private
*/
namespace detail { }


/**
@brief queries the version information in a string format @c major.minor.patch

Release notes are available here: https://taskflow.github.io/taskflow/Releases.html
*/
constexpr const char* version() {
  return "3.7.0";
}


}  // end of namespace collie::tf -----------------------------------------------------





