// Copyright 2024 The EA Authors.
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

#include <hercules/hir/transform/parallel/schedule.h>

#include <hercules/hir/ir.h>
#include <hercules/hir/util/irtools.h>

#include <cctype>
#include <sstream>

namespace hercules {
namespace ir {
namespace transform {
namespace parallel {
namespace {
int getScheduleCode(const std::string &schedule = "static", bool chunked = false,
                    bool ordered = false, bool monotonic = false) {
  // codes from "enum sched_type" at
  // https://github.com/llvm/llvm-project/blob/main/openmp/runtime/src/kmp.h
  int modifier = monotonic ? (1 << 29) : (1 << 30);
  if (schedule == "static") {
    if (chunked) {
      if (ordered)
        return 65;
      else
        return 33;
    } else {
      if (ordered)
        return 66;
      else
        return 34;
    }
  } else if (schedule == "dynamic") {
    return (ordered ? 67 : 35) | modifier;
  } else if (schedule == "guided") {
    return (ordered ? 68 : 36) | modifier;
  } else if (schedule == "runtime") {
    return (ordered ? 69 : 37) | modifier;
  } else if (schedule == "auto") {
    return (ordered ? 70 : 38) | modifier;
  }
  return getScheduleCode(); // default
}

Value *nullIfNeg(Value *v) {
  if (v && util::isConst<int64_t>(v) && util::getConst<int64_t>(v) <= 0)
    return nullptr;
  return v;
}
} // namespace

OMPSched::OMPSched(int code, bool dynamic, Value *threads, Value *chunk, bool ordered,
                   int64_t collapse, bool gpu)
    : code(code), dynamic(dynamic), threads(nullIfNeg(threads)),
      chunk(nullIfNeg(chunk)), ordered(ordered), collapse(collapse), gpu(gpu) {
  if (code < 0)
    this->code = getScheduleCode();
}

OMPSched::OMPSched(const std::string &schedule, Value *threads, Value *chunk,
                   bool ordered, int64_t collapse, bool gpu)
    : OMPSched(getScheduleCode(schedule, nullIfNeg(chunk) != nullptr, ordered),
               (schedule != "static") || ordered, threads, chunk, ordered, collapse,
               gpu) {}

std::vector<Value *> OMPSched::getUsedValues() const {
  std::vector<Value *> ret;
  if (threads)
    ret.push_back(threads);
  if (chunk)
    ret.push_back(chunk);
  return ret;
}

int OMPSched::replaceUsedValue(id_t id, Value *newValue) {
  auto count = 0;
  if (threads && threads->getId() == id) {
    threads = newValue;
    ++count;
  }
  if (chunk && chunk->getId() == id) {
    chunk = newValue;
    ++count;
  }
  return count;
}

} // namespace parallel
} // namespace transform
} // namespace ir
} // namespace hercules
