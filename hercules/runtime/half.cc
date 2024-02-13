// Copyright 2022 ByteDance Ltd. and/or its affiliates.
#include "hercules/runtime/half.h"

namespace hercules {
namespace runtime {

static_assert(std::is_standard_layout<Half>::value, "c10::Half must be standard layout.");

std::ostream& operator<<(std::ostream& out, const Half& value) {
  out << (float)value;
  return out;
}

}  // namespace runtime
}  // namespace hercules
