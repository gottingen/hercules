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

# Core library

from internal.core import *
from internal.attributes import *
from internal.types.ptr import *
from internal.types.str import *
from internal.types.int import *
from internal.types.bool import *
from internal.types.array import *
from internal.types.error import *
from internal.types.intn import *
from internal.types.float import *
from internal.types.byte import *
from internal.types.generator import *
from internal.types.optional import *
from internal.types.slice import *
from internal.types.range import *
from internal.types.complex import *
from internal.internal import *
from internal.types.strbuf import strbuf as _strbuf
from internal.types.collections.list import *
import internal.c_stubs as _C
from internal.format import *

def next(g: Generator[T], default: Optional[T] = None, T: type) -> T:
    if g.done():
        if default:
            return unwrap(default)
        else:
            raise StopIteration()
    return g.next()

from C import hs_print_full(str, cobj)

class Set:
    items: List[T]
    T: type

    def __init__(self):
        self.items = []

    def __iter__(self) -> Generator[T]:
        yield from self.items

    def add(self, what: T):
        if what not in self.items:
            self.items.append(what)

    def __repr__(self) -> str:
        s = self.items.__repr__()
        s.ptr[0] = "{".ptr[0]
        s.ptr[s.len - 1] = "}".ptr[0]
        return s

class Dict:
    keys: List[K]
    values: List[V]
    K: type
    V: type

    def __init__(self):
        self.keys = []
        self.values = []

    def __iter__(self) -> Generator[K]:
        yield from self.keys

    def items(self) -> Generator[Tuple[K, V]]:
        for i in range(self.keys.len):
            yield (self.keys[i], self.values[i])

    def __contains__(self, key: K) -> bool:
        return self.keys.find(key) != -1

    def __getitem__(self, key: K) -> V:
        i = self.keys.find(key)
        return self.values[i]

    def __setitem__(self, key: K, val: V):
        i = self.keys.find(key)
        if i != -1:
            self.values[i] = val
        else:
            self.keys.append(key)
            self.values.append(val)

    def __len__(self) -> int:
        return self.keys.len

    def __repr__(self) -> str:
        n = self.__len__()
        if n == 0:
            return "{}"
        else:
            lst = []
            lst.append("{")
            first = True
            for k, v in self.items():
                if not first:
                    lst.append(", ")
                else:
                    first = False
                lst.append(k.__repr__())
                lst.append(": ")
                lst.append(v.__repr__())
            lst.append("}")
            return str.cat(lst)

@extend
class str:
    def __getitem__(self, idx: int) -> str:
        if idx < 0:
            idx += self.len
        if not (0 <= idx < self.len):
            raise IndexError("string index out of range")
        return str(self.ptr + idx, 1)

    def __getitem__(self, s: Slice) -> str:
        if s.start is None and s.stop is None and s.step is None:
            return self.__copy__()
        elif s.step is None:
            start, stop, step, length = s.adjust_indices(self.len)
            return str(self.ptr + start, length)
        else:
            raise ValueError("nope")

    def strip(self):
        if self.__len__() == 0:
            return ""

        i = 0
        while i < self.__len__() and _C.isspace(i32(int(self.ptr[i]))):
            i += 1

        j = self.__len__() - 1
        while j >= 0 and _C.isspace(i32(int(self.ptr[j]))):
            j -= 1
        j += 1

        if j <= i:
            return ""

        return str(self.ptr + i, j - i)

    def __repr__(self) -> str:
        return f"'{self}'"

from internal.builtin import *

from internal.dlopen import dlsym as _dlsym
