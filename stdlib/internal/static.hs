#
# Copyright 2023 EA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Methods for static reflection. Implemented within call.cpp and/or loops.cpp.
# !! Not intended for public use !!

def fn_overloads(T: type, F: Static[str]):
    pass

def fn_args(F):  # function: (i, name)
    pass

def fn_arg_has_type(F, i: Static[int]):
    pass

def fn_arg_get_type(F, i: Static[int]):
    pass

def fn_can_call(F, *args, **kwargs):
    pass

def fn_wrap_call_args(F, *args, **kwargs):
    pass

def fn_has_default(F, i: Static[int]):
    pass

def fn_get_default(F, i: Static[int]):
    pass

@no_type_wrap
def static_print(*args):
    pass

def vars(obj, with_index: Static[int] = 0):
    pass

def vars_types(T: type, with_index: Static[int] = 0):
    pass

def tuple_type(T: type, N: Static[int]):
    pass
