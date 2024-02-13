// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from Halide.
 *
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
#include <hercules/runtime/data_type.h>

#include <hercules/runtime/c_runtime_api.h>
#include <hercules/runtime/logging.h>
#include <hercules/runtime/registry.h>

namespace hercules {
namespace runtime {

String GetCustomTypeName(uint8_t type_code) {
  auto f = ::hercules::runtime::FunctionRegistry::Get("runtime._datatype_get_type_name");
  HSCHECK(f) << "Function runtime._datatype_get_type_name not found";
  return (*f)({RTView(type_code)}).As<String>();
}

uint8_t GetCustomTypeCode(const String& type_name) {
  auto f = ::hercules::runtime::FunctionRegistry::Get("runtime._datatype_get_type_code");
  HSCHECK(f) << "Function runtime._datatype_get_type_code not found";
  return (*f)({RTView(type_name)}).As<int>();
}

bool GetCustomTypeRegistered(uint8_t type_code) {
  auto f = ::hercules::runtime::FunctionRegistry::Get("runtime._datatype_get_type_registered");
  HSCHECK(f) << "Function runtime._datatype_get_type_registered not found";
  return (*f)({RTView(type_code)}).As<bool>();
}

uint8_t ParseCustomDatatype(const String& s, const char** scan) {
  HSCHECK(s.substr(0, 6) == "custom") << "Not a valid custom datatype string";

  auto tmp = s.c_str();

  HSCHECK(s.c_str() == tmp);
  *scan = s.c_str() + 6;
  HSCHECK(s.c_str() == tmp);
  if (**scan != '[')
    HSLOG(FATAL) << "expected opening brace after 'custom' type in" << s;
  HSCHECK(s.c_str() == tmp);
  *scan += 1;
  HSCHECK(s.c_str() == tmp);
  size_t custom_name_len = 0;
  HSCHECK(s.c_str() == tmp);
  while (*scan + custom_name_len <= s.c_str() + s.length() && *(*scan + custom_name_len) != ']')
    ++custom_name_len;
  HSCHECK(s.c_str() == tmp);
  if (*(*scan + custom_name_len) != ']')
    HSLOG(FATAL) << "expected closing brace after 'custom' type in" << s;
  HSCHECK(s.c_str() == tmp);
  *scan += custom_name_len + 1;
  HSCHECK(s.c_str() == tmp);

  auto type_name = s.substr(7, custom_name_len);
  HSCHECK(s.c_str() == tmp);
  return GetCustomTypeCode(type_name);
}

DataType::DataType(int code, int bits, int lanes) {
  data_.code = static_cast<uint8_t>(code);
  data_.bits = static_cast<uint8_t>(bits);
  data_.lanes = static_cast<uint16_t>(lanes);
  if (code == kBFloat) {
    HSCHECK_EQ(bits, 16);
  }
}

DataType DataType::ShapeIndex() {
  if (std::is_signed<hvm_script_index_t>::value) {
    return DataType::Int(sizeof(hvm_script_index_t) * 8);
  } else {
    return DataType::UInt(sizeof(hvm_script_index_t) * 8);
  }
}

int GetVectorBytes(DataType dtype) {
  int data_bits = dtype.bits() * dtype.lanes();
  // allow bool to exist
  if (dtype == DataType::Bool() || dtype == DataType::Int(4) || dtype == DataType::UInt(4) ||
      dtype == DataType::Int(1)) {
    return 1;
  }
  HSCHECK_EQ(data_bits % 8, 0U) << "Need to load/store by multiple of bytes";
  return data_bits / 8;
}

const char* DLDataTypeCode2Str(DLDataTypeCode type_code) {
  switch (static_cast<int>(type_code)) {
    case kDLInt:
      return "int";
    case kDLUInt:
      return "uint";
    case kDLFloat:
      return "float";
    case DataType::kHandle:
      return "handle";
    case kDLBfloat:
      return "bfloat";
    default:
      HSLOG(FATAL) << "unknown type_code=" << static_cast<int>(type_code);
      return "";
  }
}

DLDataType String2DLDataType(string_view s) {
  DLDataType t;
  // handle void type
  if (s.length() == 0) {
    t = DataType::Void();
    return t;
  }
  t.bits = 32;
  t.lanes = 1;
  const char* scan;
  if (s.substr(0, 3) == "int") {
    t.code = kDLInt;
    scan = s.data() + 3;
  } else if (s.substr(0, 4) == "uint") {
    t.code = kDLUInt;
    scan = s.data() + 4;
  } else if (s.substr(0, 5) == "float") {
    t.code = kDLFloat;
    scan = s.data() + 5;
  } else if (s.substr(0, 6) == "handle") {
    t.code = DataType::kHandle;
    t.bits = 64;  // handle uses 64 bit by default.
    scan = s.data() + 6;
  } else if (s == "bool") {
    t.code = kDLUInt;
    t.bits = 1;
    t.lanes = 1;
    return t;
  } else if (s.substr(0, 6) == "bfloat") {
    t.code = DataType::kBFloat;
    scan = s.data() + 6;
  } else if (s.substr(0, 6) == "custom") {
    t.code = ParseCustomDatatype(s, &scan);
  } else {
    scan = s.data();
    HSLOG(FATAL) << "unknown type " << s;
  }
  char* xdelim;  // emulate sscanf("%ux%u", bits, lanes)
  uint8_t bits = static_cast<uint8_t>(strtoul(scan, &xdelim, 10));
  if (bits != 0)
    t.bits = bits;
  char* endpt = xdelim;
  if (*xdelim == 'x') {
    t.lanes = static_cast<uint16_t>(strtoul(xdelim + 1, &endpt, 10));
  }
  HSCHECK(endpt == s.data() + s.length()) << "unknown type " << s;
  return t;
}

std::ostream& operator<<(std::ostream& os, DLDataType t) {  // NOLINT(*)
  if (t.bits == 1 && t.lanes == 1 && t.code == kDLUInt) {
    os << "bool";
    return os;
  }
  if (DataType(t).is_void()) {
    return os << "void";
  }
  if (t.code < DataType::kCustomBegin) {
    os << DLDataTypeCode2Str(static_cast<DLDataTypeCode>(t.code));
  } else {
    os << "custom[" << GetCustomTypeName(t.code) << "]";
  }
  if (t.code == DataType::kHandle)
    return os;
  os << static_cast<int>(t.bits);
  if (t.lanes != 1) {
    os << 'x' << static_cast<int>(t.lanes);
  }
  return os;
}

String DLDataType2String(DLDataType t) {
  if (t.bits == 0)
    return {};
  std::ostringstream os;
  os << t;
  return os.str();
}

}  // namespace runtime
}  // namespace hercules