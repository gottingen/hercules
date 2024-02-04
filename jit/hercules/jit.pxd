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

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "hercules/compiler/jit_extern.h" namespace "hercules::jit":
    cdef cppclass JIT
    cdef cppclass JITResult:
        void *result
        string message
        bint operator bool()

    JIT *jitInit(string)
    JITResult jitExecuteSafe(JIT*, string, string, int, char)
    JITResult jitExecutePython(JIT*, string, vector[string], string, vector[string], object, char)
    string getJITLibrary()
