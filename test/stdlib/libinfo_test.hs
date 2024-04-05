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
from libinfo import *

@test
def test_lib_info():
    hello = lib_echo_hello("world")
    assert len(hello) == 13
    print(hello) #: Hello, world!
    jeff = lib_echo_hello("Jeff")
    assert len(jeff) == 12
    print(jeff) #: Hello, Jeff!

test_lib_info()

@test
def test_find_library():
    lib = find_fly()
    libfly = find_lib("fly")
    assert lib == libfly
    print(lib) #: /usr/local/lib/libfly.so
    nill = find_lib("nill")
    assert nill == None

test_find_library()

@test
def test_find_include():
    inc = find_inc("flare.h")
    assert inc == "/usr/local/include/flare.h"
    print(inc) #: /usr/local/include/flare.h

test_find_include()