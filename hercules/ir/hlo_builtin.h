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
 * \file hvm/ir/hlo_builtin.h
 * \brief high level ir builtin intrinsics.
 *
 */
#pragma once

#include <hercules/ir/op_expr.h>

namespace hercules {
namespace ir {

/*! \brief Collection of builtin intrinsics as ops */
namespace builtin {
/******************************************************************************
 * make_kwargs_op
 *****************************************************************************/

HERCULES_DLL const Op& make_kwargs_op();

/******************************************************************************
 * torch ops
 *****************************************************************************/

HERCULES_DLL const Op& torch_ops();
/******************************************************************************
 * numpy ops
 *****************************************************************************/
HERCULES_DLL const Op& numpy_ops();

/******************************************************************************
 * lambda function
 *****************************************************************************/
HERCULES_DLL const Op& call_lambda();

/******************************************************************************
 * builtins functions
 *****************************************************************************/
HERCULES_DLL const Op& hlo_if_then_else();

/******************************************************************************
 * List builtin functions
 *****************************************************************************/
HERCULES_DLL const Op& list___len__();
HERCULES_DLL const Op& list___contains__();
HERCULES_DLL const Op& list___getitem__();
HERCULES_DLL const Op& list___setitem__();
HERCULES_DLL const Op& list___getslice__();
HERCULES_DLL const Op& list_append();
HERCULES_DLL const Op& list_extend();
HERCULES_DLL const Op& list_repeat();
HERCULES_DLL const Op& list_fused_repeat_one();
HERCULES_DLL const Op& list_fused_repeat_many();
HERCULES_DLL const Op& list_reserve();
HERCULES_DLL const Op& list_index();
HERCULES_DLL const Op& list_capacity();
HERCULES_DLL const Op& list_pop();
HERCULES_DLL const Op& list_insert();
HERCULES_DLL const Op& list_remove();
HERCULES_DLL const Op& list_clear();
HERCULES_DLL const Op& list_reverse();
HERCULES_DLL const Op& list_count();
HERCULES_DLL const Op& list_sort_no_key();
HERCULES_DLL const Op& list_sort();

HERCULES_DLL const Op& ft_list___len__();
HERCULES_DLL const Op& ft_list___contains__();
HERCULES_DLL const Op& ft_list___getitem__();
HERCULES_DLL const Op& ft_list___setitem__();
HERCULES_DLL const Op& ft_list___getslice__();
HERCULES_DLL const Op& ft_list_append();
HERCULES_DLL const Op& ft_list_extend();
HERCULES_DLL const Op& ft_list_repeat();
HERCULES_DLL const Op& ft_list_fused_repeat_one();
HERCULES_DLL const Op& ft_list_fused_repeat_many();
HERCULES_DLL const Op& ft_list_reserve();
HERCULES_DLL const Op& ft_list_index();
HERCULES_DLL const Op& ft_list_capacity();
HERCULES_DLL const Op& ft_list_pop();
HERCULES_DLL const Op& ft_list_insert();
HERCULES_DLL const Op& ft_list_remove();
HERCULES_DLL const Op& ft_list_clear();
HERCULES_DLL const Op& ft_list_reverse();
HERCULES_DLL const Op& ft_list_count();
HERCULES_DLL const Op& ft_list_sort_no_key();
HERCULES_DLL const Op& ft_list_sort();
/******************************************************************************
 * Dict builtin functions
 *****************************************************************************/
HERCULES_DLL const Op& dict___len__();
HERCULES_DLL const Op& dict___contains__();
HERCULES_DLL const Op& dict___getitem__();
HERCULES_DLL const Op& dict___setitem__();
HERCULES_DLL const Op& dict_clear();
HERCULES_DLL const Op& dict_reserve();
HERCULES_DLL const Op& dict_bucket_count();
HERCULES_DLL const Op& dict_keys();
HERCULES_DLL const Op& dict_values();
HERCULES_DLL const Op& dict_items();
HERCULES_DLL const Op& dict_get();
HERCULES_DLL const Op& dict_pop();

HERCULES_DLL const Op& ft_dict___len__();
HERCULES_DLL const Op& ft_dict___contains__();
HERCULES_DLL const Op& ft_dict___getitem__();
HERCULES_DLL const Op& ft_dict___setitem__();
HERCULES_DLL const Op& ft_dict_clear();
HERCULES_DLL const Op& ft_dict_reserve();
HERCULES_DLL const Op& ft_dict_bucket_count();
HERCULES_DLL const Op& ft_dict_keys();
HERCULES_DLL const Op& ft_dict_values();
HERCULES_DLL const Op& ft_dict_items();
HERCULES_DLL const Op& ft_dict_get();
HERCULES_DLL const Op& ft_dict_pop();
/******************************************************************************
 * ADT builtin functions
 *****************************************************************************/
HERCULES_DLL const Op& tuple_len();
/******************************************************************************
 * Set builtin functions
 *****************************************************************************/
HERCULES_DLL const Op& set___len__();
HERCULES_DLL const Op& set___contains__();
HERCULES_DLL const Op& set_add();
HERCULES_DLL const Op& set_clear();
HERCULES_DLL const Op& set_reserve();
HERCULES_DLL const Op& set_bucket_count();
HERCULES_DLL const Op& set_difference();
HERCULES_DLL const Op& set_difference_update();
HERCULES_DLL const Op& set_update();
HERCULES_DLL const Op& set_union();
HERCULES_DLL const Op& set_discard();

HERCULES_DLL const Op& ft_set___len__();
HERCULES_DLL const Op& ft_set___contains__();
HERCULES_DLL const Op& ft_set_add();
HERCULES_DLL const Op& ft_set_clear();
HERCULES_DLL const Op& ft_set_reserve();
HERCULES_DLL const Op& ft_set_bucket_count();
HERCULES_DLL const Op& ft_set_difference();
HERCULES_DLL const Op& ft_set_difference_update();
HERCULES_DLL const Op& ft_set_update();
HERCULES_DLL const Op& ft_set_union();
HERCULES_DLL const Op& ft_set_discard();
/******************************************************************************
 * String builtin functions
 *****************************************************************************/
HERCULES_DLL const Op& str_lower();
HERCULES_DLL const Op& str_upper();
HERCULES_DLL const Op& str_append();
HERCULES_DLL const Op& str_decode();
/******************************************************************************
 * Unicode builtin functions
 *****************************************************************************/
HERCULES_DLL const Op& unicode_find();
HERCULES_DLL const Op& unicode_encode();

/******************************************************************************
 * NDArray builtin functions
 *****************************************************************************/
HERCULES_DLL const Op& ndarray___getitem__();
HERCULES_DLL const Op& ndarray_getitem_as_int64();
HERCULES_DLL const Op& ndarray_getitem_as_double();
HERCULES_DLL const Op& ndarray___setitem__();
HERCULES_DLL const Op& ndarray_fused_getitem();
HERCULES_DLL const Op& ndarray_fused_getitem_as_int64();
HERCULES_DLL const Op& ndarray_fused_getitem_as_double();
HERCULES_DLL const Op& ndarray_fused_setitem();

/******************************************************************************
 * Fused functions
 *****************************************************************************/
HERCULES_DLL const Op& str_fused_concat();
HERCULES_DLL const Op& unicode_fused_concat();

/******************************************************************************
 * UserData dispatch
 *****************************************************************************/
HERCULES_DLL const Op& object___getitem__();
HERCULES_DLL const Op& object___setitem__();
HERCULES_DLL const Op& object___fused_getitem__();
HERCULES_DLL const Op& object___fused_setitem__();
HERCULES_DLL const Op& object___dispatch__();
HERCULES_DLL const Op& object___getattr__();
HERCULES_DLL const Op& object___setattr__();
HERCULES_DLL const Op& user_data_get_attr();
HERCULES_DLL const Op& user_data_set_attr();
HERCULES_DLL const Op& user_data_call();
HERCULES_DLL const Op& user_data_call_attr();
/******************************************************************************
 * Generic Container builtin functions
 *****************************************************************************/
HERCULES_DLL const Op& object_append();
HERCULES_DLL const Op& object_slice_append();
HERCULES_DLL const Op& object_contains();
HERCULES_DLL const Op& object_slice_contains();
HERCULES_DLL const Op& object_add();
HERCULES_DLL const Op& object_extend();
HERCULES_DLL const Op& object_slice_add();
HERCULES_DLL const Op& object_clear();
HERCULES_DLL const Op& object_slice_clear();
HERCULES_DLL const Op& object_get_item();
HERCULES_DLL const Op& object_slice_get_item();
HERCULES_DLL const Op& object_set_item();
HERCULES_DLL const Op& object_slice_set_item();

HERCULES_DLL const Op& object_slice_load();
HERCULES_DLL const Op& object_slice_store();

HERCULES_DLL const Op& object_find();

HERCULES_DLL const Op& object_slice_lower();
HERCULES_DLL const Op& object_slice_upper();
HERCULES_DLL const Op& object_slice_isdigit();
HERCULES_DLL const Op& object_slice_isalpha();

HERCULES_DLL const Op& builtins_print();
HERCULES_DLL const Op& object_call();

HERCULES_DLL const Op& builtins_unpack();

}  // namespace builtin
}  // namespace ir
}  // namespace hercules
