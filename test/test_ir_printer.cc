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
#include <hercules/codegen/codegen_c_host.h>
#include <hercules/ir/function.h>
#include <hercules/ir/module.h>
#include <hercules/ir/prim_builtin.h>
#include <hercules/ir/prim_expr.h>
#include <hercules/runtime/container.h>
#include <hercules/runtime/logging.h>
#include <hercules/runtime/registry.h>
#include <iostream>

#include <gtest/gtest.h>

namespace hercules {
namespace ir {
using namespace runtime;

TEST(IR, Printer) {
  const auto* printer = ::hercules::runtime::FunctionRegistry::Get("node.IRTextPrinter_Print");
  const auto* build_module = ::hercules::runtime::FunctionRegistry::Get("module.build.c");

  PrimExpr a(3);
  PrimExpr b(4);

  PrimAdd c(a, b);

  PrimMul d(c, a);

  Bool cond(true);

  PrimCall custom(d.dtype(), builtin::if_then_else(), {cond, d, c});

  Array<Stmt> seq_stmt;
  seq_stmt.push_back(ExprStmt(custom));
  SeqStmt body(seq_stmt);
  Array<PrimVar> params{PrimVar("n", DataType::Bool())};
  PrimFunc func(params, {}, body, PrimType(DataType::Int(32)));
  func = WithAttr(std::move(func), attr::kGlobalSymbol, StringRef("test_arith"));

  StringRef ir_text = (*printer)({func, None}).As<StringRef>();
  std::cout << ir_text << std::endl;

  codegen::CodeGenCHost cg;
  cg.AddFunction(func);
  std::string code = cg.Finish();
  std::cout << code << std::endl;

  IRModule mod;
  mod->Add(func);
  ::hercules::runtime::Module m = (*build_module)({mod}).As<Module>();
  std::cout << m->GetSource() << std::endl;
}

}  // namespace ir
}  // namespace hercules
