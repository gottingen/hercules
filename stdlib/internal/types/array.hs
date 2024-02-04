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

from internal.gc import sizeof

@extend
class Array:
    def __new__(ptr: Ptr[T], sz: int) -> Array[T]:
        return (sz, ptr)

    def __new__(sz: int) -> Array[T]:
        return (sz, Ptr[T](sz))

    def __copy__(self) -> Array[T]:
        p = Ptr[T](self.len)
        str.memcpy(p.as_byte(), self.ptr.as_byte(), self.len * sizeof(T))
        return (self.len, p)

    def __deepcopy__(self) -> Array[T]:
        p = Ptr[T](self.len)
        i = 0
        while i < self.len:
            p[i] = self.ptr[i].__deepcopy__()
            i += 1
        return (self.len, p)

    def __len__(self) -> int:
        return self.len

    def __bool__(self) -> bool:
        return bool(self.len)

    def __getitem__(self, index: int) -> T:
        return self.ptr[index]

    def __setitem__(self, index: int, what: T):
        self.ptr[index] = what

    def slice(self, s: int, e: int) -> Array[T]:
        return (e - s, self.ptr + s)

array = Array
