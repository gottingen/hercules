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
#include <hercules/ir/prim_expr.h>

#include <hercules/ir/_base/reflection.h>
#include <hercules/ir/op_attr_types.h>
#include <hercules/ir/printer/ir_docsifier.h>
#include <hercules/runtime/container.h>
#include <hercules/runtime/functor.h>
#include <hercules/runtime/registry.h>

namespace hercules {
namespace ir {

using namespace ::hercules::runtime;
using namespace ::hercules::ir::printer;

PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm(runtime::DataType::Int(32), value)) {
}

PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(runtime::DataType::Float(32), value)) {
}

IntImm::IntImm(runtime::DataType dtype, int64_t value, Span span) {
  HSCHECK(dtype.is_scalar()) << "ValueError: IntImm can only take scalar.";
  HSCHECK(dtype.is_int() || dtype.is_uint())
      << "ValueError: IntImm supports only int or uint type.";
  if (dtype.is_uint()) {
    HSCHECK_GE(value, 0U);
  }
  ObjectPtr<IntImmNode> node = runtime::make_object<IntImmNode>();
  node->dtype = dtype;
  node->checked_type_ = PrimType(node->dtype);
  node->value = value;
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_NODE_TYPE(IntImmNode);

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IntImm>("", [](IntImm s, ObjectPath p, IRDocsifier d) -> Doc {
      return LiteralDoc::Int(s->value, p->Attr("value"));
    });

FloatImm::FloatImm(runtime::DataType dtype, double value, Span span) {
  HSCHECK_EQ(dtype.lanes(), 1) << "ValueError: FloatImm can only take scalar.";
  ObjectPtr<FloatImmNode> node = runtime::make_object<FloatImmNode>();
  node->dtype = dtype;
  node->checked_type_ = PrimType(node->dtype);
  node->value = value;
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_NODE_TYPE(FloatImmNode);

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<FloatImm>("", [](FloatImm s, ObjectPath p, IRDocsifier d) -> Doc {
      return LiteralDoc::Float(s->value, p->Attr("value"));
    });

// PrimCast
PrimCast::PrimCast(DataType t, PrimExpr value, Span span) {
  HSCHECK(value.defined());
  HSCHECK_EQ(t.lanes(), value.dtype().lanes());
  ObjectPtr<PrimCastNode> node = make_object<PrimCastNode>();
  node->dtype = t;
  node->checked_type_ = PrimType(node->dtype);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimCast")
    .set_body_typed([](DataType dtype, PrimExpr value, Span span) {
      return PrimCast(dtype, std::move(value), std::move(span));
    });

HERCULES_REGISTER_NODE_TYPE(PrimCastNode);

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PrimCast>("", [](PrimCast s, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc value = d->AsDoc<ExprDoc>(s->value, p->Attr("value"));
      if (s->dtype == runtime::DataType::Int(64)) {
        return IdDoc("int")->Call({value});
      } else if (s->dtype == runtime::DataType::Float(64)) {
        return IdDoc("float")->Call({value});
      }
      if (d->cfg->ignore_type_cast) {
        return value;
      }
      ExprDoc dtype = LiteralDoc::DataType(s->dtype, p->Attr("dtype"));
      return Dialect(d, "PrimCast")->Call({dtype, value});
    });

// HLOCastPrim
HLOCastPrim::HLOCastPrim(DataType t, BaseExpr value, Span span) {
  HSCHECK(value.defined());
  ObjectPtr<HLOCastPrimNode> node = make_object<HLOCastPrimNode>();
  node->dtype = t;
  node->checked_type_ = PrimType(node->dtype);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.HLOCastPrim")
    .set_body_typed([](DataType dtype, BaseExpr value, Span span) {
      return HLOCastPrim(dtype, value);
    });

HERCULES_REGISTER_NODE_TYPE(HLOCastPrimNode);

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<HLOCastPrim>("", [](HLOCastPrim s, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc value = d->AsDoc<ExprDoc>(s->value, p->Attr("value"));
      if (d->cfg->ignore_type_cast) {
        return value;
      }
      ExprDoc dtype = LiteralDoc::DataType(s->dtype, p->Attr("dtype"));
      return Dialect(d, "HLOCastPrim")->Call({dtype, value});
    });

#define HERCULES_DEFINE_BINOP_CONSTRUCTOR(Name)                       \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                       \
    using T = Name::ContainerType;                                      \
    HSCHECK(a.defined()) << "ValueError: a is undefined\n";             \
    HSCHECK(b.defined()) << "ValueError: b is undefined\n";             \
    HSCHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n"; \
    ObjectPtr<T> node = make_object<T>();                               \
    node->dtype = a.dtype();                                            \
    node->checked_type_ = PrimType(node->dtype);                        \
    node->a = std::move(a);                                             \
    node->b = std::move(b);                                             \
    node->span = std::move(span);                                       \
    data_ = std::move(node);                                            \
  }

#define HERCULES_DEFINE_CMPOP_CONSTRUCTOR(Name)                       \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                       \
    using T = Name::ContainerType;                                      \
    HSCHECK(a.defined()) << "ValueError: a is undefined\n";             \
    HSCHECK(b.defined()) << "ValueError: b is undefined\n";             \
    HSCHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n"; \
    ObjectPtr<T> node = make_object<T>();                               \
    node->dtype = DataType::Bool(a.dtype().lanes());                    \
    node->checked_type_ = PrimType(node->dtype);                        \
    node->a = std::move(a);                                             \
    node->b = std::move(b);                                             \
    node->span = std::move(span);                                       \
    data_ = std::move(node);                                            \
  }

#define HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(                                          \
    NodeType, NodeObj, NodeFunc, OpString, OpKind)                                                \
  HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                               \
      .set_dispatch<ir::NodeType>("", [](ir::NodeType node, ObjectPath p, IRDocsifier d) -> Doc { \
        ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));                                     \
        ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));                                     \
        return OperationDoc(OperationDocNode::Kind::OpKind, {a, b});                              \
      });

#define HERCULES_SCRIPT_PRINTER_DEF_BINARY(NodeType, OpString)                                  \
  HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)                                               \
      .set_dispatch<ir::NodeType>("", [](ir::NodeType node, ObjectPath p, IRDocsifier d) -> Doc { \
        ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));                                     \
        ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));                                     \
        return Dialect(d, OpString)->Call({a, b});                                                \
      });

// PrimAdd
HERCULES_DEFINE_BINOP_CONSTRUCTOR(PrimAdd);

HERCULES_REGISTER_GLOBAL("ir.PrimAdd").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimAdd(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimAddNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimAdd, PrimAddNode, add, "Add", kAdd);

// PrimSub
HERCULES_DEFINE_BINOP_CONSTRUCTOR(PrimSub);

HERCULES_REGISTER_GLOBAL("ir.PrimSub").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimSub(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimSubNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimSub, PrimSubNode, sub, "Sub", kSub);

// PrimMul
HERCULES_DEFINE_BINOP_CONSTRUCTOR(PrimMul);

HERCULES_REGISTER_GLOBAL("ir.PrimMul").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimMul(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimMulNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimMul, PrimMulNode, mul, "Mul", kMult);

// PrimDiv
HERCULES_DEFINE_BINOP_CONSTRUCTOR(PrimDiv);

HERCULES_REGISTER_GLOBAL("ir.PrimDiv").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimDiv(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimDivNode);

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PrimDiv>("", [](PrimDiv node, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
      ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));
      PrimExpr ret = hercules::ir::div(node->a, node->b);
      if (!ret->IsInstance<PrimDivNode>()) {
        return Dialect(d, "PrimDiv")->Call({a, b});
      }
      if ((node->a->dtype.is_int() || node->a->dtype.is_uint()) &&
          (node->b->dtype.is_int() || node->b->dtype.is_uint())) {
        return Dialect(d, "PrimDiv")->Call({a, b});
      }
      return OperationDoc(OperationDocNode::Kind::kDiv, {a, b});
    });

// PrimMod
HERCULES_DEFINE_BINOP_CONSTRUCTOR(PrimMod);

HERCULES_REGISTER_GLOBAL("ir.PrimMod").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimMod(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimModNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY(PrimMod, "truncmod");

// PrimFloorDiv
PrimFloorDiv::PrimFloorDiv(PrimExpr a, PrimExpr b, Span span) {
  using T = PrimFloorDiv::ContainerType;
  HSCHECK(a.defined()) << "ValueError: a is undefined\n";
  HSCHECK(b.defined()) << "ValueError: b is undefined\n";

  bool a_is_int = a.dtype().is_int() || a.dtype().is_uint();
  bool b_is_int = b.dtype().is_int() || b.dtype().is_uint();
  bool is_both_int = a_is_int && b_is_int;
  if (!is_both_int) {
    a = cast(DataType::Float(64), std::move(a));
    b = cast(DataType::Float(64), std::move(b));
  }
  ObjectPtr<T> node = make_object<T>();
  node->dtype = a.dtype();
  node->checked_type_ = PrimType(node->dtype);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimFloorDiv").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimFloorDiv(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimFloorDivNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(
    PrimFloorDiv, PrimFloorDivNode, floordiv, "FloorDiv", kFloorDiv);

// PrimFloorMod
PrimFloorMod::PrimFloorMod(PrimExpr a, PrimExpr b, Span span) {
  using T = PrimFloorMod::ContainerType;
  HSCHECK(a.defined()) << "ValueError: a is undefined\n";
  HSCHECK(b.defined()) << "ValueError: b is undefined\n";

  bool a_is_int = a.dtype().is_int() || a.dtype().is_uint();
  bool b_is_int = b.dtype().is_int() || b.dtype().is_uint();
  bool is_both_int = a_is_int && b_is_int;
  if (!is_both_int) {
    a = cast(DataType::Float(64), std::move(a));
    b = cast(DataType::Float(64), std::move(b));
  }
  ObjectPtr<T> node = make_object<T>();
  node->dtype = a.dtype();
  node->checked_type_ = PrimType(node->dtype);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimFloorMod").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimFloorMod(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimFloorModNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(
    PrimFloorMod, PrimFloorModNode, floormod, "FloorMod", kMod);

// PrimMin
HERCULES_DEFINE_BINOP_CONSTRUCTOR(PrimMin);

HERCULES_REGISTER_GLOBAL("ir.PrimMin").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimMin(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimMinNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY(PrimMin, "min");

// PrimMax
HERCULES_DEFINE_BINOP_CONSTRUCTOR(PrimMax);

HERCULES_REGISTER_GLOBAL("ir.PrimMax").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimMax(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimMaxNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY(PrimMax, "max");

// PrimEQ
HERCULES_DEFINE_CMPOP_CONSTRUCTOR(PrimEQ);

HERCULES_REGISTER_GLOBAL("ir.PrimEQ").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimEQ(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimEQNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimEQ, PrimEQNode, equal, "EQ", kEq);

// PrimNE
HERCULES_DEFINE_CMPOP_CONSTRUCTOR(PrimNE);

HERCULES_REGISTER_GLOBAL("ir.PrimNE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimNE(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimNENode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimNE, PrimNENode, not_equal, "NE", kNotEq);

// PrimLT
HERCULES_DEFINE_CMPOP_CONSTRUCTOR(PrimLT);

HERCULES_REGISTER_GLOBAL("ir.PrimLT").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimLT(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimLTNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimLT, PrimLTNode, less_than, "LT", kLt);

// PrimLE
HERCULES_DEFINE_CMPOP_CONSTRUCTOR(PrimLE);

HERCULES_REGISTER_GLOBAL("ir.PrimLE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimLE(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimLENode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimLE, PrimLENode, less_or_equal, "LE", kLtE);

// PrimGT
HERCULES_DEFINE_CMPOP_CONSTRUCTOR(PrimGT);

HERCULES_REGISTER_GLOBAL("ir.PrimGT").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimGT(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimGTNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimGT, PrimGTNode, greater_than, "GT", kGt);

// PrimGE
HERCULES_DEFINE_CMPOP_CONSTRUCTOR(PrimGE);

HERCULES_REGISTER_GLOBAL("ir.PrimGE").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimGE(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimGENode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimGE, PrimGENode, greater_or_equal, "GE", kGtE);

// PrimAnd
PrimAnd::PrimAnd(PrimExpr a, PrimExpr b, Span span) {
  HSCHECK(a.defined()) << "ValueError: a is undefined";
  HSCHECK(b.defined()) << "ValueError: b is undefined";
  HSCHECK(a.dtype().is_bool() || a.dtype().is_int());
  HSCHECK(b.dtype().is_bool() || b.dtype().is_int());

  ObjectPtr<PrimAndNode> node = make_object<PrimAndNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->checked_type_ = PrimType(node->dtype);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimAnd").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimAnd(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimAndNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimAnd, PrimAndNode, logic_and, "And", kAnd);

// PrimOr
PrimOr::PrimOr(PrimExpr a, PrimExpr b, Span span) {
  HSCHECK(a.defined()) << "ValueError: a is undefined";
  HSCHECK(b.defined()) << "ValueError: b is undefined";
  HSCHECK(a.dtype().is_bool() || a.dtype().is_int());
  HSCHECK(b.dtype().is_bool() || b.dtype().is_int());

  ObjectPtr<PrimOrNode> node = make_object<PrimOrNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->checked_type_ = PrimType(node->dtype);
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimOr").set_body_typed([](PrimExpr a, PrimExpr b, Span span) {
  return PrimOr(std::move(a), std::move(b), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimOrNode);

HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(PrimOr, PrimOrNode, logic_or, "Or", kOr);

#undef HERCULES_SCRIPT_PRINTER_DEF_BINARY
#undef HERCULES_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR

// PrimNot
PrimNot::PrimNot(PrimExpr a, Span span) {
  HSCHECK(a.defined()) << "ValueError: a is undefined";
  HSCHECK(a.dtype().is_bool() || a.dtype().is_int());

  ObjectPtr<PrimNotNode> node = make_object<PrimNotNode>();
  node->dtype = DataType::Bool(a.dtype().lanes());
  node->checked_type_ = PrimType(node->dtype);
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimNot").set_body_typed([](PrimExpr a, Span span) {
  return PrimNot(std::move(a), span);
});

HERCULES_REGISTER_NODE_TYPE(PrimNotNode);

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::PrimNot>("", [](ir::PrimNot node, ObjectPath p, IRDocsifier d) -> Doc {
      ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
      return OperationDoc(OperationDocNode::Kind::kNot, {a});
    });

// PrimSelect
PrimSelect::PrimSelect(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
  HSCHECK(condition.defined()) << "ValueError: condition is undefined";
  HSCHECK(true_value.defined()) << "ValueError: true_value is undefined";
  HSCHECK(false_value.defined()) << "ValueError: true_value is undefined";
  HSCHECK(condition.dtype().is_bool());
  HSCHECK(condition.dtype().lanes() == true_value.dtype().lanes() ||
          condition.dtype().lanes() == 1);
  HSCHECK(false_value.dtype() == true_value.dtype()) << "TypeError: mismatched types";

  ObjectPtr<PrimSelectNode> node = make_object<PrimSelectNode>();
  node->dtype = true_value.dtype();
  node->checked_type_ = PrimType(node->dtype);
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimSelect")
    .set_body_typed([](PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
      return PrimSelect(std::move(condition), std::move(true_value), std::move(false_value), span);
    });

HERCULES_REGISTER_NODE_TYPE(PrimSelectNode);

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::PrimSelect>(
        "", [](ir::PrimSelect select, ObjectPath p, IRDocsifier d) -> Doc {
          return Dialect(d, "PrimSelect")
              ->Call({
                  d->AsDoc<ExprDoc>(select->condition, p->Attr("condition")),
                  d->AsDoc<ExprDoc>(select->true_value, p->Attr("true_value")),
                  d->AsDoc<ExprDoc>(select->false_value, p->Attr("false_value")),
              });
        });

// Let
PrimLet::PrimLet(PrimVar var, PrimExpr value, PrimExpr body, Span span) {
  HSCHECK(value.defined());
  HSCHECK(body.defined());
  HSCHECK(var.as<PrimExprNode>());
  HSCHECK_EQ(value.dtype(), var.as<PrimExprNode>()->dtype);

  ObjectPtr<PrimLetNode> node = make_object<PrimLetNode>();
  node->dtype = body.dtype();
  node->checked_type_ = PrimType(node->dtype);
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimLet")
    .set_body_typed([](PrimVar var, PrimExpr value, PrimExpr body, Span span) {
      return PrimLet(var, value, body, span);
    });

HERCULES_REGISTER_NODE_TYPE(PrimLetNode);

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::PrimLet>("", [](ir::PrimLet let, ObjectPath p, IRDocsifier d) -> Doc {
      DictDoc where({d->AsDoc<ExprDoc>(let->var, p->Attr("var"))},
                    {d->AsDoc<ExprDoc>(let->value, p->Attr("value"))});
      return Dialect(d, "PrimLet")
          ->Call({d->AsDoc<ExprDoc>(let->body, p->Attr("body"))},  //
                 {"where"},
                 {where});
    });

// Call
PrimCall::PrimCall(DataType dtype, HLOExpr op, Array<PrimExpr> args, Span span) {
  for (size_t i = 0; i < args.size(); ++i) {
    HSCHECK(args[i].defined());
  }

  ObjectPtr<PrimCallNode> node = make_object<PrimCallNode>();
  node->dtype = dtype;
  node->checked_type_ = PrimType(node->dtype);
  node->op = std::move(op);
  node->args = std::move(args);
  node->span = std::move(span);
  data_ = std::move(node);
}

HERCULES_REGISTER_GLOBAL("ir.PrimCall")
    .set_body_typed([](DataType type, HLOExpr op, Array<ObjectRef> args, Span span) {
      Array<PrimExpr> prim_expr_args;
      for (const auto& it : args) {
        HSCHECK(it->IsInstance<PrimExprNode>());
        prim_expr_args.push_back(Downcast<PrimExpr>(it));
      }
      return PrimCall(type, op, prim_expr_args, span);
    });

HERCULES_REGISTER_NODE_TYPE(PrimCallNode);

template <typename DocType, typename AST>
static inline Array<DocType> build_arrays(const Array<AST>& ast_list,
                                          ObjectPath p,
                                          IRDocsifier d,
                                          int start_pos) {
  Array<DocType> results;
  int n_args = ast_list.size();
  results.reserve(n_args);
  for (int i = start_pos; i < n_args; ++i) {
    results.push_back(d->AsDoc<DocType>(ast_list[i], p->ArrayIndex(i)));
  }
  return results;
};

static Doc PrimCallFunctionToDoc(StringRef fn_name,
                                 ir::PrimCall call,
                                 ObjectPath p,
                                 IRDocsifier d) {
  Array<StringRef> kw_keys;
  Array<ExprDoc> kw_values;
  runtime::string_view builtins("builtins.");
  if (runtime::StringHelper::StartsWith(fn_name, builtins)) {
    fn_name = fn_name.view().substr(builtins.size());
  }
  Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, p, d, 0);
  return IdDoc(fn_name)->Call(args, kw_keys, kw_values);
}

static Doc PrimCallMethodToDoc(StringRef method_name,
                               ir::PrimCall call,
                               ObjectPath p,
                               IRDocsifier d) {
  Array<StringRef> kw_keys;
  Array<ExprDoc> kw_values;
  HSCHECK(call->args.size() >= 1) << "internal error";
  auto self = d->AsDoc<ExprDoc>(call->args[0], p->Attr("args")->ArrayIndex(0));
  int arg_pos = 1;
  Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, p, d, arg_pos);
  return self->Attr(method_name)->Call(args, kw_keys, kw_values);
}

HERCULES_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<ir::PrimCall>("", [](ir::PrimCall call, ObjectPath call_p, IRDocsifier d) -> Doc {
      ExprDoc prefix{nullptr};
      if (const auto* op = call->op.as<OpNode>()) {
        static OpAttrMap<TPrinterGlobalSymbol> op_global_symbol =
            Op::GetAttrMap<TPrinterGlobalSymbol>("TPrinterGlobalSymbol");
        static OpAttrMap<TPrinterMethodSymbol> op_method_symbol =
            Op::GetAttrMap<TPrinterMethodSymbol>("TPrinterMethodSymbol");

        auto op_ref = GetRef<Op>(op);
        if (op_global_symbol.count(op_ref)) {
          StringRef name = op_global_symbol[op_ref];
          return PrimCallFunctionToDoc(name, call, call_p, d);
        } else if (op_method_symbol.count(op_ref)) {
          StringRef name = op_method_symbol[op_ref];
          return PrimCallMethodToDoc(name, call, call_p, d);
        } else {
          StringRef name = op->name;
          prefix = Dialect(d, name);
        }
      } else if (const auto* gv = call->op.as<GlobalVarNode>()) {
        prefix = LiteralDoc::Str(gv->name_hint, call_p->Attr("op"));
      } else {
        prefix = d->AsDoc<ExprDoc>(call->op, call_p->Attr("op"));
      }
      Array<ExprDoc> args = build_arrays<ExprDoc>(call->args, call_p, d, 0);
      return prefix->Call(args);
    });

HERCULES_REGISTER_GLOBAL("runtime.GetIntImm").set_body_typed([](IntImm i) { return i->value; });

HERCULES_REGISTER_GLOBAL("runtime.GetFloatImm").set_body_typed([](FloatImm f) {
  return f->value;
});

}  // namespace ir
}  // namespace hercules
