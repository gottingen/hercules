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
# Primarily for internal use. Regular users should not use this module.

@pure
@C
def hs_alloc(a: int) -> cobj:
    pass

@pure
@C
def hs_alloc_atomic(a: int) -> cobj:
    pass

@pure
@C
def hs_alloc_uncollectable(a: int) -> cobj:
    pass

@pure
@C
def hs_alloc_atomic_uncollectable(a: int) -> cobj:
    pass

@nocapture
@derives
@C
def hs_realloc(p: cobj, newsize: int, oldsize: int) -> cobj:
    pass

@nocapture
@C
def hs_free(p: cobj) -> None:
    pass

@nocapture
@C
def hs_register_finalizer(p: cobj, f: cobj) -> None:
    pass

@nocapture
@C
def hs_gc_add_roots(p: cobj, q: cobj) -> None:
    pass

@nocapture
@C
def hs_gc_remove_roots(p: cobj, q: cobj) -> None:
    pass

@C
def hs_gc_clear_roots() -> None:
    pass

@nocapture
@C
def hs_gc_exclude_static_roots(p: cobj, q: cobj) -> None:
    pass

def sizeof(T: type):
    return T.__elemsize__

def atomic(T: type):
    return T.__atomic__

def alloc(sz: int):
    return hs_alloc(sz)

# Allocates a block of memory via GC, where the
# caller guarantees that this block will not store
# pointers to other GC-allocated data.
def alloc_atomic(sz: int):
    return hs_alloc_atomic(sz)

# Allocates a block of memory via GC that is scanned,
# but not collected itself. Should be free'd explicitly.
def alloc_uncollectable(sz: int):
    return hs_alloc_uncollectable(sz)

# Allocates a block of memory via GC that is scanned,
# but not collected itself. Should be free'd explicitly.
def alloc_atomic_uncollectable(sz: int):
    return hs_alloc_atomic_uncollectable(sz)

def realloc(p: cobj, newsz: int, oldsz: int):
    return hs_realloc(p, newsz, oldsz)

def free(p: cobj):
    hs_free(p)

def add_roots(start: cobj, end: cobj):
    hs_gc_add_roots(start, end)

def remove_roots(start: cobj, end: cobj):
    hs_gc_remove_roots(start, end)

def clear_roots():
    hs_gc_clear_roots()

def exclude_static_roots(start: cobj, end: cobj):
    hs_gc_exclude_static_roots(start, end)

def register_finalizer(p):
    if hasattr(p, "__del__"):

        def f(x: cobj, data: cobj, T: type):
            Ptr[T](__ptr__(x).as_byte())[0].__del__()

        hs_register_finalizer(p.__raw__(), f(T=type(p), ...).__raw__())

def construct_ref[T](args) -> T:
    p = T.__new__()
    p.__init__(*args)
    return p
