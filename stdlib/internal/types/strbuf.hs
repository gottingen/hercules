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

class strbuf:
    data: Ptr[byte]
    n: int
    m: int

    def __init__(self, capacity: int = 16):
        self.data = Ptr[byte](capacity)
        self.n = 0
        self.m = capacity

    def append(self, s: str):
        from internal.gc import realloc
        adding = s.__len__()
        needed = self.n + adding
        if needed > self.m:
            m = self.m
            while m < needed:
                m *= 2
            self.data = realloc(self.data, m, self.m)
            self.m = m
        str.memcpy(self.data + self.n, s.ptr, adding)
        self.n = needed

    def reverse(self):
        a = 0
        b = self.n - 1
        p = self.data
        while a < b:
            p[a], p[b] = p[b], p[a]
            a += 1
            b -= 1

    def __str__(self):
        return str(self.data, self.n)
