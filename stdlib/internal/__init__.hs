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

from internal.attributes import *
from internal.static import static_print as __static_print__
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

__argv__ = Array[str](0)

from internal.types.strbuf import strbuf as _strbuf
from internal.types.collections.list import *
from internal.types.collections.set import *
from internal.types.collections.dict import *
from internal.types.collections.tuple import *

# Extended core library

import internal.c_stubs as _C
from internal.format import *
from internal.builtin import *
from internal.builtin import _jit_display
from internal.str import *

from internal.sort import sorted

from openmp import Ident as __OMPIdent, for_par
from gpu import _gpu_loop_outline_template
from internal.file import File, gzFile, open, gzopen
from pickle import pickle, unpickle
from internal.dlopen import dlsym as _dlsym
import internal.python

if __py_numerics__:
    import internal.pynumerics
if __py_extension__:
    internal.python.ensure_initialized()
