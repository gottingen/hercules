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

def __ac_isempty(flag: Ptr[u32], i: int) -> int:
    return int(flag[i >> 4] >> u32((i & 0xF) << 1)) & 2

def __ac_isdel(flag: Ptr[u32], i: int) -> int:
    return int(flag[i >> 4] >> u32((i & 0xF) << 1)) & 1

def __ac_iseither(flag: Ptr[u32], i: int) -> int:
    return int(flag[i >> 4] >> u32((i & 0xF) << 1)) & 3

def __ac_set_isdel_false(flag: Ptr[u32], i: int):
    flag[i >> 4] &= u32(~(1 << ((i & 0xF) << 1)))

def __ac_set_isempty_false(flag: Ptr[u32], i: int):
    flag[i >> 4] &= u32(~(2 << ((i & 0xF) << 1)))

def __ac_set_isboth_false(flag: Ptr[u32], i: int):
    flag[i >> 4] &= u32(~(3 << ((i & 0xF) << 1)))

def __ac_set_isdel_true(flag: Ptr[u32], i: int):
    flag[i >> 4] |= u32(1 << ((i & 0xF) << 1))

def __ac_fsize(m) -> int:
    return 1 if m < 16 else m >> 4
