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

@pure
@C
def seq_is_macos() -> bool:
    pass

# <dlfcn.h>
from C import dlerror() -> cobj as c_dlerror
from C import dlopen(cobj, int) -> cobj as c_dlopen
from C import dlsym(cobj, cobj) -> cobj as c_dlsym
from C import dlclose(cobj) -> i32 as c_dlclose

RTLD_NOW = 2
RTLD_GLOBAL = 8 if seq_is_macos() else 256
RTLD_LOCAL = 0 if seq_is_macos() else 256

def dlext() -> str:
    if seq_is_macos():
        return "dylib"
    else:
        return "so"

@pure
def dlerror() -> str:
    return str.from_ptr(c_dlerror())

def dlopen(name: str, flag: int = RTLD_NOW | RTLD_GLOBAL) -> cobj:
    h = c_dlopen(cobj() if name == "" else name.c_str(), flag)
    if h == cobj():
        raise CError(dlerror())
    return h

def dlsym(lib, name: str, Fn: type) -> Fn:
    h = cobj()
    if isinstance(lib, str):
        h = dlopen(lib)
    else:
        h = lib
    fn = c_dlsym(h, name.c_str())
    if fn == cobj():
        raise CError(dlerror())
    return Fn(fn)

def dlclose(handle: cobj):
    if c_dlclose(handle) != i32(0):
        raise CError(dlerror())
