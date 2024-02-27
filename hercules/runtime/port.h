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
#pragma once


#define HS_FLAG_DEBUG (1 << 0)          // compiled/running in debug mode
#define HS_FLAG_CAPTURE_OUTPUT (1 << 1) // capture writes to stdout/stderr
#define HS_FLAG_STANDALONE (1 << 2)     // compiled as a standalone object/binary

#define HS_FUNC extern "C"

