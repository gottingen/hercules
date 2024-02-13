// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the expressions is inspired by TVM.
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
#include <hercules/ir/expr.h>
#include <hercules/ir/hlo_ops.h>

namespace hercules {
namespace ir {

/******************************************************************************
 * HLOExpr arith functions
 *****************************************************************************/
// TODO: TryConstFold
// add
BaseExpr add(BaseExpr a, BaseExpr b, Span span) {
  return HLOAdd(std::move(a), std::move(b), std::move(span));
}

// sub
BaseExpr sub(BaseExpr a, BaseExpr b, Span span) {
  return HLOSub(std::move(a), std::move(b), std::move(span));
}

// mul
BaseExpr mul(BaseExpr a, BaseExpr b, Span span) {
  return HLOMul(std::move(a), std::move(b), std::move(span));
}

// floordiv
BaseExpr floordiv(BaseExpr a, BaseExpr b, Span span) {
  return HLOFloorDiv(std::move(a), std::move(b), std::move(span));
}

// floormod
BaseExpr floormod(BaseExpr a, BaseExpr b, Span span) {
  return HLOFloorMod(std::move(a), std::move(b), std::move(span));
}

// operator>
BaseExpr greater_than(BaseExpr a, BaseExpr b, Span span) {
  return HLOGreaterThan(std::move(a), std::move(b), std::move(span));
}

BaseExpr greater_or_equal(BaseExpr a, BaseExpr b, Span span) {
  return HLOGreaterEqual(std::move(a), std::move(b), std::move(span));
}

BaseExpr less_than(BaseExpr a, BaseExpr b, Span span) {
  return HLOLessThan(std::move(a), std::move(b), std::move(span));
}

BaseExpr less_or_equal(BaseExpr a, BaseExpr b, Span span) {
  return HLOLessEqual(std::move(a), std::move(b), std::move(span));
}

BaseExpr equal(BaseExpr a, BaseExpr b, Span span) {
  return HLOEqual(std::move(a), std::move(b), std::move(span));
}

BaseExpr not_equal(BaseExpr a, BaseExpr b, Span span) {
  return HLONotEqual(std::move(a), std::move(b), std::move(span));
}

BaseExpr logic_and(BaseExpr a, BaseExpr b, Span span) {
  return HLOAnd(std::move(a), std::move(b), std::move(span));
}

BaseExpr logic_or(BaseExpr a, BaseExpr b, Span span) {
  return HLOOr(std::move(a), std::move(b), std::move(span));
}

BaseExpr logic_not(BaseExpr a, Span span) {
  return HLONot(std::move(a), std::move(span));
}

BaseExpr abs(BaseExpr a, Span span) {
  StringRef op_name = "call_extern";
  Type ret_type = a->checked_type();
  Array<BaseExpr> call_args{StringImm("ArithOps::abs"), std::move(a)};
  return Call(ret_type, Op::Get("ir." + op_name), call_args, std::move(span));
}

#define HERCULES_REGISTER_MAKE_HLO_BINARY_OP(Node, Func)                                       \
  HERCULES_REGISTER_GLOBAL("ir." #Node).set_body_typed([](BaseExpr a, BaseExpr b, Span span) { \
    return (Func(std::move(a), std::move(b), std::move(span)));                                  \
  })
#define HERCULES_REGISTER_MAKE_HLO_UNARY_OP(Node, Func)                            \
  HERCULES_REGISTER_GLOBAL("ir." #Node).set_body_typed([](BaseExpr a, Span span) { \
    return (Func(std::move(a), std::move(span)));                                    \
  })

HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpAdd, add);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpSub, sub);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpMul, mul);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpFloorDiv, floordiv);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpFloorMod, floormod);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpEQ, equal);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpNE, not_equal);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpGT, greater_than);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpGE, greater_or_equal);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpLT, less_than);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpLE, less_or_equal);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpAnd, logic_and);
HERCULES_REGISTER_MAKE_HLO_BINARY_OP(_HLO_OpOr, logic_or);
HERCULES_REGISTER_MAKE_HLO_UNARY_OP(_HLO_OpNot, logic_not);
HERCULES_REGISTER_MAKE_HLO_UNARY_OP(_HLO_OpAbs, abs);

}  // namespace ir
}  // namespace hercules
