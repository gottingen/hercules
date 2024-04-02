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
#include <collie/taskflow/dsl/connection.h>
#include <collie/taskflow/dsl/type_list.h>
#include <type_traits>

namespace collie::tf {
namespace dsl {
template <typename... Links> class TaskAnalyzer {
  template <typename FROMs, typename TOs, typename = void>
  struct BuildOneToOneLink;

  template <typename... Fs, typename Ts>
  struct BuildOneToOneLink<TypeList<Fs...>, Ts> {
    using type = Concat_t<typename BuildOneToOneLink<Fs, Ts>::type...>;
  };

  template <typename F, typename... Ts>
  struct BuildOneToOneLink<F, TypeList<Ts...>,
                           std::enable_if_t<!IsTypeList_v<F>>> {
    using type = TypeList<OneToOneLink<F, Ts>...>;
  };

  template <typename Link> class OneToOneLinkSetF {
    using FromTaskList = typename Link::FromTaskList;
    using ToTaskList = typename Link::ToTaskList;

  public:
    using type = typename BuildOneToOneLink<FromTaskList, ToTaskList>::type;
  };

public:
  using AllTasks = Unique_t<
      Concat_t<typename Links::FromTaskList..., typename Links::ToTaskList...>>;
  using OneToOneLinkSet =
      Unique_t<Flatten_t<Map_t<TypeList<Links...>, OneToOneLinkSetF>>>;
};

} // namespace dsl
} // namespace collie::tf
