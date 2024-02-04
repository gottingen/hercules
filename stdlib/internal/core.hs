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

@__internal__
class __internal__:
    pass
@__internal__
class __magic__:
    pass

@tuple
@__internal__
class NoneType:
    pass

@tuple
@__internal__
@__notuple__
class bool:
    pass

@tuple
@__internal__
@__notuple__
class byte:
    pass

@tuple
@__internal__
@__notuple__
class int:
    MAX = 9223372036854775807
    pass

@tuple
@__internal__
@__notuple__
class float:
    MIN_10_EXP = -307
    pass

@tuple
@__internal__
@__notuple__
class float32:
    MIN_10_EXP = -37
    pass

@tuple
@__internal__
@__notuple__
class float16:
    MIN_10_EXP = -4
    pass

@tuple
@__internal__
@__notuple__
class bfloat16:
    MIN_10_EXP = -37
    pass

@tuple
@__internal__
@__notuple__
class float128:
    MIN_10_EXP = -4931
    pass

@tuple
@__internal__
class type:
    pass

@tuple
@__internal__
@__notuple__
class Function[T, TR]:
    pass

@tuple
@__internal__
class Callable[T, TR]:
    pass

@tuple
@__internal__
@__notuple__
class Ptr[T]:
    pass
cobj = Ptr[byte]

@tuple
@__internal__
@__notuple__
class Generator[T]:
    pass

@tuple
@__internal__
@__notuple__
class Optional:
    T: type = NoneType

@tuple
@__internal__
@__notuple__
class Int[N: Static[int]]:
    pass

@tuple
@__internal__
@__notuple__
class UInt[N: Static[int]]:
    pass

@__internal__
class pyobj:
    p: Ptr[byte]

@tuple
@__internal__
class str:
    ptr: Ptr[byte]
    len: int


@tuple
@__internal__
class Tuple:
    @__internal__
    def __new__() -> Tuple:
        pass
    def __add__(self, obj):
        return __magic__.add(self, obj)
    def __mul__(self, n: Static[int]):
        return __magic__.mul(self, n)
    def __contains__(self, obj) -> bool:
        return __magic__.contains(self, obj)
    def __getitem__(self, idx: int):
        return __magic__.getitem(self, idx)
    def __iter__(self):
        yield from __magic__.iter(self)
    def __hash__(self) -> int:
        return __magic__.hash(self)
    def __repr__(self) -> str:
        return __magic__.repr(self)
    def __len__(self) -> int:
        return __magic__.len(self)
    def __eq__(self, obj: Tuple) -> bool:
        return __magic__.eq(self, obj)
    def __ne__(self, obj: Tuple) -> bool:
        return __magic__.ne(self, obj)
    def __gt__(self, obj: Tuple) -> bool:
        return __magic__.gt(self, obj)
    def __ge__(self, obj: Tuple) -> bool:
        return __magic__.ge(self, obj)
    def __lt__(self, obj: Tuple) -> bool:
        return __magic__.lt(self, obj)
    def __le__(self, obj: Tuple) -> bool:
        return __magic__.le(self, obj)
    def __pickle__(self, dest: Ptr[byte]):
        return __magic__.pickle(self, dest)
    def __unpickle__(src: Ptr[byte]) -> Tuple:
        return __magic__.unpickle(src)
    def __to_py__(self) -> Ptr[byte]:
        return __magic__.to_py(self)
    def __from_py__(src: Ptr[byte]) -> Tuple:
        return __magic__.from_py(src)
    def __to_gpu__(self, cache) -> Tuple:
        return __magic__.to_gpu(self, cache)
    def __from_gpu__(self, other: Tuple):
        return __magic__.from_gpu(self, other)
    def __from_gpu_new__(other: Tuple) -> Tuple:
        return __magic__.from_gpu_new(other)
    def __tuplesize__(self) -> int:
        return __magic__.tuplesize(self)


@tuple
@__internal__
class Array:
    len: int
    ptr: Ptr[T]
    T: type

@extend
class type:
    def __new__(obj):
        pass
function = Function

@__internal__
class Ref[T]:
    pass

@tuple
@__internal__
@__notuple__
class Union[TU]:
    # compiler-generated
    def __new__(val):
        TU
    def __call__(self, *args, **kwargs):
        return __internal__.union_call(self, args, kwargs)

# dummy
@__internal__
class TypeVar[T]: pass
@__internal__
class ByVal: pass
@__internal__
class ByRef: pass

@__internal__
class ClassVar[T]:
    pass

@__internal__
class RTTI:
    id: int

@__internal__
@tuple
class ellipsis: pass

@tuple
@__internal__
class __array__:
    T: type
    def __new__(sz: Static[int]) -> Array[T]:
        pass

def __ptr__(var):
    pass

def staticlen(obj):
    pass

def compile_error(msg: Static[str]):
    pass

def isinstance(obj, what):
    pass

@__attribute__
def overload():
    pass

def hasattr(obj, attr: Static[str], *args, **kwargs):
    """Special handling"""
    pass

def getattr(obj, attr: Static[str]):
    pass

def setattr(obj, attr: Static[str], what):
    pass

def tuple(iterable):
    pass

def super():
    pass

def superf(*args):
    """Special handling"""
    pass

def __realized__(fn, args):
    pass

def statictuple(*args):
    return args

def __has_rtti__(T: type):
    pass
