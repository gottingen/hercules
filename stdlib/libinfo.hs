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


#from C import echo_hello() -> str
from C import say_hello(str) -> str
from C import find_library(str) -> str
from C import find_include(str) -> str

def lib_echo_hello(w: str) -> str:
    return say_hello(w)

#def lib_say_hello(name: str) -> str:
#    return say_hello(name)

def find_fly() -> str:
    return find_library("fly")

def find_lib(name: str) -> optional[str]:
    lp = find_library(name)
    if lp == "":
        return None
    return lp

def find_inc(name: str) -> optional[str]:
    inp = find_include(name)
    if inp == "":
        return None
    return inp



