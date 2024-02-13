// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
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

/*!
 * \file src/ir/prim_builtin.cc
 *
 *  builtin intrinsic operators.
 */
#include <hercules/ir/prim_builtin.h>

#include <hercules/ir/op_attr_types.h>
#include <hercules/ir/op_expr.h>
#include <hercules/runtime/registry.h>

namespace hercules {
namespace ir {
namespace builtin {

#define HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(OpName) \
  const Op& OpName() {                                 \
    static const Op& op = Op::Get("ir." #OpName);      \
    return op;                                         \
  }                                                    \
  HERCULES_IR_REGISTER_OP("ir." #OpName)

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(reinterpret)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_num_inputs(1);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(likely)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kExprAnnotation))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(bitwise_and)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(bitwise_or)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(bitwise_xor)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(bitwise_not)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(shift_left)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(shift_right)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(large_uint_imm)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(address_of)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_num_inputs(1);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(if_then_else)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(q_multiply_shift)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(isnullptr).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(isnan).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(popcount)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(fma)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(call_extern)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(call_pure_extern)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(call_llvm_intrin)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(call_llvm_pure_intrin)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(call_spirv_pure_glsl450)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(prefetch).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(vectorhigh)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(vectorlow).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

HERCULES_IR_DEFINE_PRIM_BUILTIN_FUNC(vectorcombine)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

}  // namespace builtin
}  // namespace ir
}  // namespace hercules
