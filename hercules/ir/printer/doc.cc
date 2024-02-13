// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from TVM.
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
#include <hercules/ir/printer/doc.h>

#include <hercules/runtime/logging.h>
#include <hercules/runtime/registry.h>

namespace hercules {
namespace ir {
namespace printer {

using runtime::GetRef;
using runtime::make_object;

ExprDoc ExprDocNode::Attr(StringRef attr) const {
  return AttrAccessDoc(GetRef<ExprDoc>(this), std::move(attr));
}

ExprDoc ExprDocNode::operator[](Array<Doc> indices) const {
  return IndexDoc(GetRef<ExprDoc>(this), std::move(indices));
}

ExprDoc ExprDocNode::Call(Array<ExprDoc, void> args) const {
  return CallDoc(GetRef<ExprDoc>(this), std::move(args), Array<StringRef>(), Array<ExprDoc>());
}

ExprDoc ExprDocNode::Call(Array<ExprDoc, void> args,
                          Array<StringRef, void> kwargs_keys,
                          Array<ExprDoc, void> kwargs_values) const {
  return CallDoc(
      GetRef<ExprDoc>(this), std::move(args), std::move(kwargs_keys), std::move(kwargs_values));
}

ExprDoc ExprDoc::operator[](Array<Doc> indices) const {
  return (*get())[std::move(indices)];
}

StmtBlockDoc::StmtBlockDoc(Array<StmtDoc> stmts) {
  ObjectPtr<StmtBlockDocNode> n = make_object<StmtBlockDocNode>();
  n->stmts = std::move(stmts);
  this->data_ = std::move(n);
}

LiteralDoc::LiteralDoc(ObjectRef value, const Optional<ObjectPath>& object_path) {
  ObjectPtr<LiteralDocNode> n = make_object<LiteralDocNode>();
  n->value = value;
  if (object_path.defined()) {
    n->source_paths.push_back(object_path.value());
  }
  this->data_ = std::move(n);
}

IdDoc::IdDoc(StringRef name) {
  ObjectPtr<IdDocNode> n = make_object<IdDocNode>();
  n->name = std::move(name);
  this->data_ = std::move(n);
}

AttrAccessDoc::AttrAccessDoc(ExprDoc value, StringRef name) {
  ObjectPtr<AttrAccessDocNode> n = make_object<AttrAccessDocNode>();
  n->value = std::move(value);
  n->name = std::move(name);
  this->data_ = std::move(n);
}

IndexDoc::IndexDoc(ExprDoc value, Array<Doc> indices) {
  ObjectPtr<IndexDocNode> n = make_object<IndexDocNode>();
  n->value = std::move(value);
  n->indices = std::move(indices);
  this->data_ = std::move(n);
}

CallDoc::CallDoc(ExprDoc callee,
                 Array<ExprDoc> args,
                 Array<StringRef> kwargs_keys,
                 Array<ExprDoc> kwargs_values) {
  ObjectPtr<CallDocNode> n = make_object<CallDocNode>();
  n->callee = std::move(callee);
  n->args = std::move(args);
  n->kwargs_keys = std::move(kwargs_keys);
  n->kwargs_values = std::move(kwargs_values);
  this->data_ = std::move(n);
}

OperationDoc::OperationDoc(OperationDocNode::Kind kind, Array<ExprDoc> operands) {
  ObjectPtr<OperationDocNode> n = make_object<OperationDocNode>();
  n->kind = kind;
  n->operands = std::move(operands);
  this->data_ = std::move(n);
}

LambdaDoc::LambdaDoc(Array<IdDoc> args, ExprDoc body) {
  ObjectPtr<LambdaDocNode> n = make_object<LambdaDocNode>();
  n->args = std::move(args);
  n->body = std::move(body);
  this->data_ = std::move(n);
}

TupleDoc::TupleDoc(Array<ExprDoc> elements) {
  ObjectPtr<TupleDocNode> n = make_object<TupleDocNode>();
  n->elements = std::move(elements);
  this->data_ = std::move(n);
}

ListDoc::ListDoc(Array<ExprDoc> elements) {
  ObjectPtr<ListDocNode> n = make_object<ListDocNode>();
  n->elements = std::move(elements);
  this->data_ = std::move(n);
}

DictDoc::DictDoc(Array<ExprDoc> keys, Array<ExprDoc> values) {
  ObjectPtr<DictDocNode> n = make_object<DictDocNode>();
  n->keys = std::move(keys);
  n->values = std::move(values);
  this->data_ = std::move(n);
}

SetDoc::SetDoc(Array<ExprDoc> elements) {
  ObjectPtr<SetDocNode> n = make_object<SetDocNode>();
  n->elements = std::move(elements);
  this->data_ = std::move(n);
}

ComprehensionDoc::ComprehensionDoc(ExprDoc target, ExprDoc iter, Optional<Array<ExprDoc>> ifs) {
  ObjectPtr<ComprehensionDocNode> n = make_object<ComprehensionDocNode>();
  n->target = std::move(target);
  n->iter = std::move(iter);
  n->ifs = std::move(ifs);
  this->data_ = std::move(n);
}

ListCompDoc::ListCompDoc(ExprDoc elt, Array<ComprehensionDoc> generators) {
  ObjectPtr<ListCompDocNode> n = make_object<ListCompDocNode>();
  n->elt = std::move(elt);
  n->generators = std::move(generators);
  this->data_ = std::move(n);
}

SetCompDoc::SetCompDoc(ExprDoc elt, Array<ComprehensionDoc> generators) {
  ObjectPtr<SetCompDocNode> n = make_object<SetCompDocNode>();
  n->elt = std::move(elt);
  n->generators = std::move(generators);
  this->data_ = std::move(n);
}

DictCompDoc::DictCompDoc(ExprDoc key, ExprDoc value, Array<ComprehensionDoc> generators) {
  ObjectPtr<DictCompDocNode> n = make_object<DictCompDocNode>();
  n->key = std::move(key);
  n->value = std::move(value);
  n->generators = std::move(generators);
  this->data_ = std::move(n);
}

SliceDoc::SliceDoc(Optional<ExprDoc> start, Optional<ExprDoc> stop, Optional<ExprDoc> step) {
  ObjectPtr<SliceDocNode> n = make_object<SliceDocNode>();
  n->start = std::move(start);
  n->stop = std::move(stop);
  n->step = std::move(step);
  this->data_ = std::move(n);
}

YieldDoc::YieldDoc(Optional<ExprDoc> value) {
  ObjectPtr<YieldDocNode> n = make_object<YieldDocNode>();
  n->value = std::move(value);
  this->data_ = std::move(n);
}

AssignDoc::AssignDoc(ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation) {
  HSCHECK(rhs.defined() || annotation.defined())
      << "ValueError: At least one of rhs and annotation needs to be non-null for AssignDoc.";
  HSCHECK(lhs->IsInstance<IdDocNode>() || annotation == nullptr)
      << "ValueError: annotation can only be nonnull if lhs is an identifier.";

  ObjectPtr<AssignDocNode> n = make_object<AssignDocNode>();
  n->lhs = std::move(lhs);
  n->rhs = std::move(rhs);
  n->annotation = std::move(annotation);
  this->data_ = std::move(n);
}

IfDoc::IfDoc(ExprDoc predicate, Array<StmtDoc> then_branch, Array<StmtDoc> else_branch) {
  HSCHECK(!then_branch.empty() || !else_branch.empty())
      << "ValueError: At least one of the then branch or else branch needs to be non-empty.";

  ObjectPtr<IfDocNode> n = make_object<IfDocNode>();
  n->predicate = std::move(predicate);
  n->then_branch = std::move(then_branch);
  n->else_branch = std::move(else_branch);
  this->data_ = std::move(n);
}

WhileDoc::WhileDoc(ExprDoc predicate, Array<StmtDoc> body) {
  ObjectPtr<WhileDocNode> n = make_object<WhileDocNode>();
  n->predicate = std::move(predicate);
  n->body = std::move(body);
  this->data_ = std::move(n);
}

ForDoc::ForDoc(ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
  ObjectPtr<ForDocNode> n = make_object<ForDocNode>();
  n->lhs = std::move(lhs);
  n->rhs = std::move(rhs);
  n->body = std::move(body);
  this->data_ = std::move(n);
}

BreakDoc::BreakDoc() {
  ObjectPtr<BreakDocNode> n = make_object<BreakDocNode>();
  this->data_ = std::move(n);
}

ContinueDoc::ContinueDoc() {
  ObjectPtr<ContinueDocNode> n = make_object<ContinueDocNode>();
  this->data_ = std::move(n);
}

ExceptionHandlerDoc::ExceptionHandlerDoc(Optional<ExprDoc> type,
                                         Optional<ExprDoc> name,
                                         Array<StmtDoc> body) {
  ObjectPtr<ExceptionHandlerDocNode> n = make_object<ExceptionHandlerDocNode>();
  n->type = std::move(type);
  n->name = std::move(name);
  n->body = std::move(body);
  this->data_ = std::move(n);
}

TryExceptDoc::TryExceptDoc(Array<StmtDoc> body,
                           Optional<Array<ExceptionHandlerDoc>> handlers,
                           Optional<Array<StmtDoc>> orelse,
                           Optional<Array<StmtDoc>> finalbody) {
  ObjectPtr<TryExceptDocNode> n = make_object<TryExceptDocNode>();
  n->body = std::move(body);
  n->handlers = std::move(handlers);
  n->orelse = std::move(orelse);
  n->finalbody = std::move(finalbody);
  this->data_ = std::move(n);
}

RaiseDoc::RaiseDoc(Optional<ExprDoc> exc, Optional<ExprDoc> cause) {
  ObjectPtr<RaiseDocNode> n = make_object<RaiseDocNode>();
  n->exc = std::move(exc);
  n->cause = std::move(cause);
  this->data_ = std::move(n);
}

ScopeDoc::ScopeDoc(Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body) {
  ObjectPtr<ScopeDocNode> n = make_object<ScopeDocNode>();
  n->lhs = std::move(lhs);
  n->rhs = std::move(rhs);
  n->body = std::move(body);
  this->data_ = std::move(n);
}

ScopeDoc::ScopeDoc(ExprDoc rhs, Array<StmtDoc> body) {
  ObjectPtr<ScopeDocNode> n = make_object<ScopeDocNode>();
  n->lhs = NullOpt;
  n->rhs = std::move(rhs);
  n->body = std::move(body);
  this->data_ = std::move(n);
}

ExprStmtDoc::ExprStmtDoc(ExprDoc expr) {
  ObjectPtr<ExprStmtDocNode> n = make_object<ExprStmtDocNode>();
  n->expr = std::move(expr);
  this->data_ = std::move(n);
}

AssertDoc::AssertDoc(ExprDoc test, Optional<ExprDoc> msg) {
  ObjectPtr<AssertDocNode> n = make_object<AssertDocNode>();
  n->test = std::move(test);
  n->msg = std::move(msg);
  this->data_ = std::move(n);
}

ReturnDoc::ReturnDoc(ExprDoc value) {
  ObjectPtr<ReturnDocNode> n = make_object<ReturnDocNode>();
  n->value = std::move(value);
  this->data_ = std::move(n);
}

FunctionDoc::FunctionDoc(IdDoc name,
                         Array<AssignDoc> args,
                         Array<ExprDoc> decorators,
                         Optional<ExprDoc> return_type,
                         Array<StmtDoc> body) {
  ObjectPtr<FunctionDocNode> n = make_object<FunctionDocNode>();
  n->name = std::move(name);
  n->args = std::move(args);
  n->decorators = std::move(decorators);
  n->return_type = std::move(return_type);
  n->body = std::move(body);
  this->data_ = std::move(n);
}

ClassDoc::ClassDoc(IdDoc name, IdDoc base, Array<ExprDoc> decorators, Array<StmtDoc> body) {
  ObjectPtr<ClassDocNode> n = make_object<ClassDocNode>();
  n->name = std::move(name);
  n->base = std::move(base);
  n->decorators = std::move(decorators);
  n->body = std::move(body);
  this->data_ = std::move(n);
}

CommentDoc::CommentDoc(StringRef comment) {
  ObjectPtr<CommentDocNode> n = make_object<CommentDocNode>();
  n->comment = std::move(comment);
  this->data_ = std::move(n);
}

DocStringDoc::DocStringDoc(StringRef docs) {
  ObjectPtr<DocStringDocNode> n = make_object<DocStringDocNode>();
  n->comment = std::move(docs);
  this->data_ = std::move(n);
}

ModuleDoc::ModuleDoc(Array<StmtDoc> body) {
  ObjectPtr<ModuleDocNode> n = make_object<ModuleDocNode>();
  n->body = std::move(body);
  this->data_ = std::move(n);
}

HERCULES_REGISTER_NODE_TYPE(DocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.DocSetSourcePaths")
    .set_body_typed([](Doc doc, Array<ObjectPath> source_paths) {
      doc->source_paths = std::move(source_paths);
    });

HERCULES_REGISTER_NODE_TYPE(ExprDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ExprDocAttr")
    .set_body_typed([](const ExprDoc& self, StringRef attr) {
      return self->Attr(std::move(attr));
    });
HERCULES_REGISTER_GLOBAL("ir.printer.ExprDocIndex")
    .set_body_typed([](const ExprDoc& self, Array<Doc> indices) {
      return self->operator[](std::move(indices));
    });
HERCULES_REGISTER_GLOBAL("ir.printer.ExprDocCall")
    .set_body_typed([](const ExprDoc& self,
                       Array<ExprDoc, void> args,
                       Array<StringRef> kwargs_keys,
                       Array<ExprDoc, void> kwargs_values) {
      return self->Call(std::move(args), std::move(kwargs_keys), std::move(kwargs_values));
    });

HERCULES_REGISTER_NODE_TYPE(StmtDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.StmtDocSetComment")
    .set_body_typed([](StmtDoc doc, Optional<StringRef> comment) {
      doc->comment = std::move(comment);
    });

HERCULES_REGISTER_NODE_TYPE(StmtBlockDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.StmtBlockDoc").set_body_typed([](Array<StmtDoc> stmts) {
  return StmtBlockDoc(std::move(stmts));
});

HERCULES_REGISTER_NODE_TYPE(LiteralDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.LiteralDocNone").set_body_typed(LiteralDoc::None);
HERCULES_REGISTER_GLOBAL("ir.printer.LiteralDocInt").set_body_typed(LiteralDoc::Int);
HERCULES_REGISTER_GLOBAL("ir.printer.LiteralDocBoolean").set_body_typed(LiteralDoc::Boolean);
HERCULES_REGISTER_GLOBAL("ir.printer.LiteralDocFloat").set_body_typed(LiteralDoc::Float);
HERCULES_REGISTER_GLOBAL("ir.printer.LiteralDocStr").set_body_typed(LiteralDoc::Str);

HERCULES_REGISTER_NODE_TYPE(IdDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.IdDoc").set_body_typed([](StringRef name) {
  return IdDoc(std::move(name));
});

HERCULES_REGISTER_NODE_TYPE(AttrAccessDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.AttrAccessDoc")
    .set_body_typed([](ExprDoc value, StringRef attr) {
      return AttrAccessDoc(std::move(value), std::move(attr));
    });

HERCULES_REGISTER_NODE_TYPE(IndexDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.IndexDoc")
    .set_body_typed([](ExprDoc value, Array<Doc> indices) {
      return IndexDoc(std::move(value), std::move(indices));
    });

HERCULES_REGISTER_NODE_TYPE(CallDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.CallDoc")
    .set_body_typed([](ExprDoc callee,                //
                       Array<ExprDoc> args,           //
                       Array<StringRef> kwargs_keys,  //
                       Array<ExprDoc> kwargs_values) {
      return CallDoc(
          std::move(callee), std::move(args), std::move(kwargs_keys), std::move(kwargs_values));
    });

HERCULES_REGISTER_NODE_TYPE(OperationDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.OperationDoc")
    .set_body_typed([](int32_t kind, Array<ExprDoc> operands) {
      return OperationDoc(OperationDocNode::Kind(kind), std::move(operands));
    });

HERCULES_REGISTER_NODE_TYPE(LambdaDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.LambdaDoc")
    .set_body_typed([](Array<IdDoc> args, ExprDoc body) {
      return LambdaDoc(std::move(args), std::move(body));
    });

HERCULES_REGISTER_NODE_TYPE(TupleDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.TupleDoc").set_body_typed([](Array<ExprDoc> elements) {
  return TupleDoc(std::move(elements));
});

HERCULES_REGISTER_NODE_TYPE(ListDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ListDoc").set_body_typed([](Array<ExprDoc> elements) {
  return ListDoc(std::move(elements));
});

HERCULES_REGISTER_NODE_TYPE(DictDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.DictDoc")
    .set_body_typed([](Array<ExprDoc> keys, Array<ExprDoc> values) {
      return DictDoc(std::move(keys), std::move(values));
    });

HERCULES_REGISTER_NODE_TYPE(SetDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.SetDoc").set_body_typed([](Array<ExprDoc> elements) {
  return SetDoc(std::move(elements));
});

HERCULES_REGISTER_NODE_TYPE(ComprehensionDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ComprehensionDoc")
    .set_body_typed([](ExprDoc target, ExprDoc iter, Optional<Array<ExprDoc>> ifs) {
      return ComprehensionDoc(std::move(target), std::move(iter), std::move(ifs));
    });

HERCULES_REGISTER_NODE_TYPE(ListCompDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ListCompDoc")
    .set_body_typed([](ExprDoc elt, Array<ComprehensionDoc> generators) {
      return ListCompDoc(std::move(elt), std::move(generators));
    });

HERCULES_REGISTER_NODE_TYPE(SetCompDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.SetCompDoc")
    .set_body_typed([](ExprDoc elt, Array<ComprehensionDoc> generators) {
      return SetCompDoc(std::move(elt), std::move(generators));
    });

HERCULES_REGISTER_NODE_TYPE(DictCompDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.DictCompDoc")
    .set_body_typed([](ExprDoc key, ExprDoc value, Array<ComprehensionDoc> generators) {
      return DictCompDoc(std::move(key), std::move(value), std::move(generators));
    });

HERCULES_REGISTER_NODE_TYPE(SliceDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.SliceDoc")
    .set_body_typed([](Optional<ExprDoc> start, Optional<ExprDoc> stop, Optional<ExprDoc> step) {
      return SliceDoc(std::move(start), std::move(stop), std::move(step));
    });

HERCULES_REGISTER_NODE_TYPE(YieldDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.YieldDoc").set_body_typed([](Optional<ExprDoc> value) {
  return YieldDoc(std::move(value));
});

HERCULES_REGISTER_NODE_TYPE(AssignDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.AssignDoc")
    .set_body_typed([](ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation) {
      return AssignDoc(std::move(lhs), std::move(rhs), std::move(annotation));
    });

HERCULES_REGISTER_NODE_TYPE(IfDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.IfDoc")
    .set_body_typed([](ExprDoc predicate, Array<StmtDoc> then_branch, Array<StmtDoc> else_branch) {
      return IfDoc(std::move(predicate), std::move(then_branch), std::move(else_branch));
    });

HERCULES_REGISTER_NODE_TYPE(WhileDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.WhileDoc")
    .set_body_typed([](ExprDoc predicate, Array<StmtDoc> body) {
      return WhileDoc(std::move(predicate), std::move(body));
    });

HERCULES_REGISTER_NODE_TYPE(ForDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ForDoc")
    .set_body_typed([](ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
      return ForDoc(std::move(lhs), std::move(rhs), std::move(body));
    });

HERCULES_REGISTER_NODE_TYPE(ContinueDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ContinueDoc").set_body_typed([]() {
  static ContinueDoc c;
  return c;
});

HERCULES_REGISTER_NODE_TYPE(BreakDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.BreakDoc").set_body_typed([]() {
  static BreakDoc c;
  return c;
});

HERCULES_REGISTER_NODE_TYPE(ExceptionHandlerDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ExceptionHandlerDoc")
    .set_body_typed([](Optional<ExprDoc> type, Optional<ExprDoc> name, Array<StmtDoc> body) {
      return ExceptionHandlerDoc(std::move(type), std::move(name), std::move(body));
    });

HERCULES_REGISTER_NODE_TYPE(TryExceptDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.TryExceptDoc")
    .set_body_typed([](Array<StmtDoc> body,
                       Optional<Array<ExceptionHandlerDoc>> handlers,
                       Optional<Array<StmtDoc>> orelse,
                       Optional<Array<StmtDoc>> finalbody) {
      return TryExceptDoc(
          std::move(body), std::move(handlers), std::move(orelse), std::move(finalbody));
    });

HERCULES_REGISTER_NODE_TYPE(RaiseDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.RaiseDoc")
    .set_body_typed([](Optional<ExprDoc> exc, Optional<ExprDoc> cause) {
      return RaiseDoc(std::move(exc), std::move(cause));
    });

HERCULES_REGISTER_NODE_TYPE(ScopeDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ScopeDoc")
    .set_body_typed([](Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body) {
      return ScopeDoc(std::move(lhs), std::move(rhs), std::move(body));
    });

HERCULES_REGISTER_NODE_TYPE(ExprStmtDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ExprStmtDoc").set_body_typed([](ExprDoc expr) {
  return ExprStmtDoc(std::move(expr));
});

HERCULES_REGISTER_NODE_TYPE(AssertDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.AssertDoc")
    .set_body_typed([](ExprDoc test, Optional<ExprDoc> msg = NullOpt) {
      return AssertDoc(std::move(test), std::move(msg));
    });

HERCULES_REGISTER_NODE_TYPE(ReturnDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ReturnDoc").set_body_typed([](ExprDoc value) {
  return ReturnDoc(std::move(value));
});

HERCULES_REGISTER_NODE_TYPE(FunctionDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.FunctionDoc")
    .set_body_typed([](IdDoc name,
                       Array<AssignDoc> args,
                       Array<ExprDoc> decorators,
                       Optional<ExprDoc> return_type,
                       Array<StmtDoc> body) {
      return FunctionDoc(std::move(name),
                         std::move(args),
                         std::move(decorators),
                         std::move(return_type),
                         std::move(body));
    });

HERCULES_REGISTER_NODE_TYPE(ClassDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ClassDoc")
    .set_body_typed([](IdDoc name, IdDoc base, Array<ExprDoc> decorators, Array<StmtDoc> body) {
      return ClassDoc(std::move(name), std::move(base), std::move(decorators), std::move(body));
    });

HERCULES_REGISTER_NODE_TYPE(CommentDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.CommentDoc").set_body_typed([](StringRef comment) {
  return CommentDoc(std::move(comment));
});

HERCULES_REGISTER_NODE_TYPE(DocStringDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.DocStringDoc").set_body_typed([](StringRef docs) {
  return DocStringDoc(std::move(docs));
});

HERCULES_REGISTER_NODE_TYPE(ModuleDocNode);
HERCULES_REGISTER_GLOBAL("ir.printer.ModuleDoc").set_body_typed([](Array<StmtDoc> body) {
  return ModuleDoc(std::move(body));
});

}  // namespace printer
}  // namespace ir
}  // namespace hercules
