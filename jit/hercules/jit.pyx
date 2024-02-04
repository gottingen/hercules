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

# distutils: language=c++
# cython: language_level=3
# cython: c_string_type=unicode
# cython: c_string_encoding=utf8

from libcpp.string cimport string
from libcpp.vector cimport vector
cimport hercules.jit


class JITError(Exception):
    pass


cdef class JITWrapper:
    cdef hercules.jit.JIT* jit

    def __cinit__(self):
        self.jit = hercules.jit.jitInit(b"hercules jit")

    def __dealloc__(self):
        del self.jit

    def execute(self, code: str, filename: str, fileno: int, debug: char) -> str:
        result = hercules.jit.jitExecuteSafe(self.jit, code, filename, fileno, <char>debug)
        if <bint>result:
            return None
        else:
            raise JITError(result.message)

    def run_wrapper(self, name: str, types: list[str], module: str, pyvars: list[str], args, debug: char) -> object:
        cdef vector[string] types_vec = types
        cdef vector[string] pyvars_vec = pyvars
        result = hercules.jit.jitExecutePython(
            self.jit, name, types_vec, module, pyvars_vec, <object>args, <char>debug
        )
        if <bint>result:
            return <object>result.result
        else:
            raise JITError(result.message)

def hercules_library():
    return hercules.jit.getJITLibrary()
