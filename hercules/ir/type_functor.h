// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the expressions is inspired by Halide/TVM IR.
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
 * \file hvm/ir/type_functor.h
 * \brief A way to defined arbitrary function signature with dispatch on types.
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <hercules/ir/adt.h>
#include <hercules/ir/base.h>
#include <hercules/runtime/container.h>
#include <hercules/runtime/functor.h>

namespace hercules {
namespace ir {

using ::hercules::runtime::NodeFunctor;

template <typename FType>
class TypeFunctor;

// functions to be overridden.
#define HERCULES_TYPE_FUNCTOR_DEFAULT \
  { return VisitTypeDefault_(op, std::forward<Args>(args)...); }

#define HERCULES_TYPE_FUNCTOR_DISPATCH(OP)                                               \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitType_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class TypeFunctor<R(const Type& n, Args...)> {
 private:
  using TSelf = TypeFunctor<R(const Type& n, Args...)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~TypeFunctor() {
  }
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Type& n, Args... args) {
    return VisitType(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitType(const Type& n, Args... args) {
    HSCHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitType_(const TypeVarNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeConstraintNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FuncTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const RangeTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TupleTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const GlobalTypeVarNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const PrimTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const PointerTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;

  virtual R VisitType_(const ObjectTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const UnicodeTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const StringTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ListTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const DictTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const SetTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const IteratorTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ExceptionTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FileTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ShapeTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const DynTensorTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ClassTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const UserDataTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const OpaqueObjectTypeNode* op,
                       Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const RefTypeNode* op, Args... args) HERCULES_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitTypeDefault_(const Object* op, Args...) {
    HSLOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;  // unreachable, written to stop compiler warning
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    HERCULES_TYPE_FUNCTOR_DISPATCH(TypeVarNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(TypeConstraintNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(FuncTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(RangeTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(TupleTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(GlobalTypeVarNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(PrimTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(PointerTypeNode);

    HERCULES_TYPE_FUNCTOR_DISPATCH(ObjectTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(UnicodeTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(StringTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(ListTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(DictTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(SetTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(IteratorTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(ExceptionTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(FileTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(ShapeTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(DynTensorTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(ClassTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(UserDataTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(OpaqueObjectTypeNode);
    HERCULES_TYPE_FUNCTOR_DISPATCH(RefTypeNode);
    return vtable;
  }
};

#undef HERCULES_TYPE_FUNCTOR_DISPATCH

/*!
 * \brief A type visitor that recursively visit types.
 */
class HERCULES_DLL TypeVisitor : public TypeFunctor<void(const Type& n)> {
 public:
  void VisitType_(const TypeVarNode* op) override;
  void VisitType_(const FuncTypeNode* op) override;
  void VisitType_(const RangeTypeNode* op) override;
  void VisitType_(const TupleTypeNode* op) override;
  void VisitType_(const GlobalTypeVarNode* op) override;
  void VisitType_(const PrimTypeNode* op) override;
  void VisitType_(const PointerTypeNode* op) override;

  void VisitType_(const ObjectTypeNode* op) override;
  void VisitType_(const UnicodeTypeNode* op) override;
  void VisitType_(const StringTypeNode* op) override;
  void VisitType_(const ListTypeNode* op) override;
  void VisitType_(const DictTypeNode* op) override;
  void VisitType_(const SetTypeNode* op) override;
  void VisitType_(const ExceptionTypeNode* op) override;
  void VisitType_(const IteratorTypeNode* op) override;
  void VisitType_(const FileTypeNode* op) override;
  void VisitType_(const ShapeTypeNode* op) override;
  void VisitType_(const DynTensorTypeNode* op) override;
  void VisitType_(const ClassTypeNode* op) override;
  void VisitType_(const UserDataTypeNode* op) override;
  void VisitType_(const OpaqueObjectTypeNode* op) override;
  void VisitType_(const RefTypeNode* op) override;
};

/*!
 * \brief TypeMutator that mutates expressions.
 */
class HERCULES_DLL TypeMutator : public TypeFunctor<Type(const Type& n)> {
 public:
  Type VisitType(const Type& t) override;
  Type VisitType_(const TypeVarNode* op) override;
  Type VisitType_(const FuncTypeNode* op) override;
  Type VisitType_(const RangeTypeNode* op) override;
  Type VisitType_(const TupleTypeNode* op) override;
  Type VisitType_(const GlobalTypeVarNode* op) override;
  Type VisitType_(const PrimTypeNode* op) override;
  Type VisitType_(const PointerTypeNode* op) override;

  Type VisitType_(const ObjectTypeNode* op) override;
  Type VisitType_(const UnicodeTypeNode* op) override;
  Type VisitType_(const StringTypeNode* op) override;
  Type VisitType_(const ListTypeNode* op) override;
  Type VisitType_(const DictTypeNode* op) override;
  Type VisitType_(const SetTypeNode* op) override;
  Type VisitType_(const ExceptionTypeNode* op) override;
  Type VisitType_(const IteratorTypeNode* op) override;
  Type VisitType_(const FileTypeNode* op) override;
  Type VisitType_(const ShapeTypeNode* op) override;
  Type VisitType_(const DynTensorTypeNode* op) override;
  Type VisitType_(const ClassTypeNode* op) override;
  Type VisitType_(const UserDataTypeNode* op) override;
  Type VisitType_(const OpaqueObjectTypeNode* op) override;
  Type VisitType_(const RefTypeNode* op) override;

 private:
  Array<Type> MutateArray(Array<Type> arr);
};

/*!
 * \brief Bind free type variables in the type.
 * \param type The type to be updated.
 * \param args_map The binding map.
 */
Type Bind(const Type& type, const Map<TypeVar, Type>& args_map);

}  // namespace ir
}  // namespace hercules
