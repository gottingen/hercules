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
// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include <collie/taskflow/core/flow_builder.h>
#include <collie/taskflow/core/task.h>
#include <collie/taskflow/dsl/type_list.h>
#include <type_traits>

namespace collie::tf {
namespace dsl {
struct TaskSignature {};

template <typename TASK, typename CONTEXT> struct TaskCb {
  using TaskType = TASK;
  void build(FlowBuilder &build, const CONTEXT &context) {
    task_ = build.emplace(TaskType{context}());
  }

  Task task_;
};

template <typename TASK> struct IsTask {
  template <typename TaskCb> struct apply {
    constexpr static bool value =
        std::is_same<typename TaskCb::TaskType, TASK>::value;
  };
};

template <typename TASK, typename = void> struct TaskTrait;

template <typename... TASK> struct SomeTask {
  using TaskList =
      Unique_t<Flatten_t<TypeList<typename TaskTrait<TASK>::TaskList...>>>;
};

// a task self
template <typename TASK>
struct TaskTrait<
    TASK, std::enable_if_t<std::is_base_of<TaskSignature, TASK>::value>> {
  using TaskList = TypeList<TASK>;
};

template <typename... TASK> struct TaskTrait<SomeTask<TASK...>> {
  using TaskList = typename SomeTask<TASK...>::TaskList;
};
} // namespace dsl
} // namespace collie::tf
