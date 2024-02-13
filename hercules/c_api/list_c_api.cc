// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <hercules/runtime/container.h>
#include <hercules/runtime/container/list_helper.h>
#include <hercules/runtime/container_private.h>
#include <hercules/runtime/ft_container.h>
#include <hercules/runtime/registry.h>

namespace hercules {
namespace runtime {

/******************************************************************************
 * List container
 *****************************************************************************/

HERCULES_REGISTER_GLOBAL("runtime.List").set_body([](PyArgs args) -> RTValue {
  List data;
  for (int i = 0; i < args.size(); ++i) {
    data.push_back(args[i].As<RTValue>());
  }
  return data;
});

HERCULES_REGISTER_GLOBAL("runtime.FTList").set_body([](PyArgs args) -> RTValue {
  FTList<RTValue> data;
  for (int i = 0; i < args.size(); ++i) {
    data.push_back(args[i].As<RTValue>());
  }
  return data;
});

HERCULES_REGISTER_GLOBAL("runtime.ListEqual").set_body([](PyArgs args) -> RTValue {
  HSCHECK(2 == args.size()) << "list.__eq__ expect " << 2 << " arguments but get " << args.size();
  RTValue lhs = args[0].As<RTValue>();
  RTValue rhs = args[1].As<RTValue>();
  return lhs == rhs;
});

HERCULES_REGISTER_GLOBAL("runtime.ListGetSlice").set_body([](PyArgs args) -> RTValue {
  HSCHECK(4 == args.size()) << "list.__getslice__ expect " << 4 << " arguments but get "
                            << args.size();
  int64_t start = args[1].As<int64_t>();
  int64_t stop = args[2].As<int64_t>();
  int64_t step = args[3].As<int64_t>();
  auto const& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      return self.AsObjectRefNoCheck<List>().get_slice(start, stop, step);
    } break;
    case TypeIndex::kRuntimeFTList: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__getslice__",
                                                                       {start, stop, step});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.List_Iter").set_body([](PyArgs args) -> RTValue {
  List container = args[0].As<List>();
  return container.iter();
});

HERCULES_REGISTER_GLOBAL("runtime.ListGetItem").set_body([](PyArgs args) -> RTValue {
  HSCHECK(2 == args.size()) << "list.__getitem__ expect " << 2 << " arguments but get "
                            << args.size();
  int64_t i = args[1].As<int64_t>();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      return self.AsObjectRefNoCheck<List>()[i];
    } break;
    case TypeIndex::kRuntimeFTList: {
      return self.AsObjectRef<FTObjectBase>().generic_call_attr("__getitem__", {i});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListSetItem").set_body([](PyArgs args) -> RTValue {
  HSCHECK(3 == args.size()) << "list.__setitem__ expect " << 3 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  int64_t i = args[1].As<int64_t>();
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().set_item(i, args[2].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__setitem__",
                                                                {i, args[2].As<RTView>()});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListSize").set_body([](PyArgs args) -> RTValue {
  HSCHECK(1 == args.size()) << "list.__len__ expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      return self.AsObjectRefNoCheck<List>().size();
    } break;
    case TypeIndex::kRuntimeFTList: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__len__", {});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListAppend").set_body([](PyArgs args) -> RTValue {
  HSCHECK(2 == args.size()) << "list.append expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().push_back(args[1].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("append", {args[1].As<RTView>()});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListExtend").set_body([](PyArgs args) -> RTValue {
  HSCHECK(2 == args.size()) << "list.extend expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().extend(args[1].AsObjectRef<List>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("extend", {args[1].As<RTView>()});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListRepeat").set_body([](PyArgs args) -> RTValue {
  HSCHECK(2 == args.size()) << "list.__mul__ expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  int64_t times = args[1].As<int64_t>();
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      return self.AsObjectRefNoCheck<List>().repeat(times);
    } break;
    case TypeIndex::kRuntimeFTList: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__mul__", {times});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListReserve").set_body([](PyArgs args) -> RTValue {
  HSCHECK(2 == args.size()) << "list.reserve expect " << 2 << " arguments but get " << args.size();
  int64_t i = args[1].As<int64_t>();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().reserve(i);
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("reserve", {i});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListCapacity").set_body([](PyArgs args) -> RTValue {
  List data = args[0].As<List>();
  return static_cast<int64_t>(data.capacity());
});

HERCULES_REGISTER_GLOBAL("runtime.ListConcat").set_body([](PyArgs args) -> RTValue {
  List data = args[0].As<List>();
  List value = args[1].As<List>();
  return List::Concat(data, value);
});

HERCULES_REGISTER_GLOBAL("runtime.ListContains").set_body([](PyArgs args) -> RTValue {
  HSCHECK(2 == args.size()) << "list.__contains__ expect " << 2 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      return self.AsObjectRefNoCheck<List>().contains(args[1].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__contains__",
                                                                       {args[1].As<RTView>()});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListSetSlice").set_body([](PyArgs args) -> RTValue {
  HSCHECK(4 == args.size()) << "list.__setslice__ expect " << 4 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  int64_t start = args[1].As<int64_t>();
  int64_t end = args[2].As<int64_t>();
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().set_slice(start, end, args[3].AsObjectRef<List>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("__setslice__",
                                                                {start, end, args[3].As<RTView>()});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListPop").set_body([](PyArgs args) -> RTValue {
  HSCHECK(1 == args.size() || 2 == args.size())
      << "list.pop expect " << 1 << " or " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      if (args.size() == 1) {
        return self.AsObjectRefNoCheck<List>().pop();
      } else {
        return self.AsObjectRefNoCheck<List>().pop(args[1].As<int64_t>());
      }
    } break;
    case TypeIndex::kRuntimeFTList: {
      if (args.size() == 1) {
        return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("pop", {});
      } else {
        return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("pop",
                                                                         {args[1].As<RTView>()});
      }
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListInsert").set_body([](PyArgs args) -> RTValue {
  HSCHECK(3 == args.size()) << "list.insert expect " << 3 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().insert(args[1].As<int64_t>(), args[2].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr(
          "insert", {args[1].As<int64_t>(), args[2].As<RTValue>()});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListRemove").set_body([](PyArgs args) -> RTValue {
  HSCHECK(2 == args.size()) << "list.remove expect " << 2 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().remove(args[1].As<RTValue>());
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("remove", {args[1].As<RTView>()});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListClear").set_body([](PyArgs args) -> RTValue {
  HSCHECK(1 == args.size()) << "list.clear expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().clear();
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("clear", {});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListReverse").set_body([](PyArgs args) -> RTValue {
  HSCHECK(1 == args.size()) << "list.reverse expect " << 1 << " arguments but get " << args.size();
  const auto& self = args[0];
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      self.AsObjectRefNoCheck<List>().reverse();
    } break;
    case TypeIndex::kRuntimeFTList: {
      self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("reverse", {});
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return None;
});

HERCULES_REGISTER_GLOBAL("runtime.ListIndex").set_body([](PyArgs args) -> RTValue {
  HSCHECK(4 <= args.size()) << "list.index expect at least " << 4 << " arguments but get "
                            << args.size();
  const auto& self = args[0];
  RTValue x = args[1].As<RTValue>();
  switch (args[0].type_code()) {
    case TypeIndex::kRuntimeList: {
      switch (args.size()) {
        case 2: {
          return self.AsObjectRefNoCheck<List>().index(std::move(x));
        } break;
        case 3: {
          return self.AsObjectRefNoCheck<List>().index(std::move(x), args[2].As<int64_t>());
        } break;
        case 4: {
          return self.AsObjectRefNoCheck<List>().index(
              std::move(x), args[2].As<int64_t>(), args[3].As<int64_t>());
        } break;
      }
    } break;
    case TypeIndex::kRuntimeFTList: {
      PyArgs newArgs(args.begin() + 1, args.size() - 1);
      return self.AsObjectRefNoCheck<FTObjectBase>().generic_call_attr("index", newArgs);
    } break;
    default: {
      HSTHROW << "expect 'list' but get '" << TypeIndex2Str(args[0].type_code());
    } break;
  }
  return -1;
});

HERCULES_REGISTER_GLOBAL("runtime.ListSort").set_body([](PyArgs args) -> RTValue {
  List data = args[0].As<List>();
  if (args.size() == 1) {
    ListHelper::Sort(data);
  } else {
    UserDataRef comp = args[1].As<UserDataRef>();
    ListHelper::Sort(data, comp);
  }
  return None;
});

}  // namespace runtime
}  // namespace hercules
