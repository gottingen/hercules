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

def system(cmd: str) -> int:
    return _C.system(cmd.c_str())

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

@tuple
class EnvMap:
    _map: Dict[str, str]

    def __new__() -> EnvMap:
        return (Dict[str, str](),)

    def _init_if_needed(self):
        if len(self._map) == 0:
            env = _C.seq_env()
            p = env[0]
            i = 0
            while p:
                s = str.from_ptr(p)
                if s:
                    j = 0
                    found = False
                    while j < len(s):
                        if s[j] == "=":
                            found = True
                            break
                        j += 1
                    k = s[0:j] if found else s
                    v = s[j + 1 :] if found else ""
                    self._map[k] = v
                i += 1
                p = env[i]

    def __getitem__(self, key: str) -> str:
        self._init_if_needed()
        return self._map[key]

    def __repr__(self) -> str:
        self._init_if_needed()
        return repr(self._map)

    def __contains__(self, key: str) -> bool:
        self._init_if_needed()
        return key in self._map

    def __iter__(self) -> Generator[Tuple[str, str]]:
        self._init_if_needed()
        return self._map.items()

environ = EnvMap()

def getenv(key: str, default: str = "") -> str:
    return environ[key] if key in environ else default

def mkdir(name: str, mode: int = 0x1FF) -> int:
    # TODO: use errno
    from C import mkdir(cobj, int) -> int
    ret = mkdir(name.ptr, mode)
    if ret != 0:
        raise OSError("mkdir failed")
