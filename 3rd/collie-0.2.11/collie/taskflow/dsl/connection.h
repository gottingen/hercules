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
#include <collie/taskflow/dsl/task_trait.h>
#include <collie/taskflow/dsl/tuple_utils.h>
#include <collie/taskflow/dsl/type_list.h>

namespace collie::tf {
namespace dsl {
template <typename F, typename T> class Connection {
  using FROMs = typename TaskTrait<F>::TaskList;
  using TOs = typename TaskTrait<T>::TaskList;

public:
  using FromTaskList = Unique_t<Flatten_t<FROMs>>;
  using ToTaskList = Unique_t<Flatten_t<TOs>>;
};

template <typename T, typename OUT = TypeList<>> struct Chain;

template <typename F, typename OUT> struct Chain<auto (*)(F)->void, OUT> {
  using From = F;
  using type = OUT;
};

template <typename F, typename T, typename OUT>
struct Chain<auto (*)(F)->T, OUT> {
private:
  using To = typename Chain<T, OUT>::From;

public:
  using From = F;
  using type = typename Chain<
      T, typename OUT::template appendTo<Connection<From, To>>>::type;
};

template <typename FROM, typename TO> struct OneToOneLink {
  template <typename TasksCB> struct InstanceType {
    constexpr void build(TasksCB &tasksCb) {
      constexpr size_t TasksCBSize = std::tuple_size<TasksCB>::value;
      constexpr size_t FromTaskIndex =
          TupleElementByF_v<TasksCB, IsTask<FROM>::template apply>;
      constexpr size_t ToTaskIndex =
          TupleElementByF_v<TasksCB, IsTask<TO>::template apply>;
      static_assert(FromTaskIndex < TasksCBSize && ToTaskIndex < TasksCBSize,
                    "fatal: not find TaskCb in TasksCB");
      std::get<FromTaskIndex>(tasksCb).task_.precede(
          std::get<ToTaskIndex>(tasksCb).task_);
    }
  };
};
} // namespace dsl
}; // namespace collie::tf
