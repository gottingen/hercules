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

@tuple
class range:
    start: int
    stop: int
    step: int

    # Magic methods

    def __new__(start: int, stop: int, step: int) -> range:
        if step == 0:
            raise ValueError("range() step argument must not be zero")
        return (start, stop, step)

    def __new__(start: int, stop: int) -> range:
        return (start, stop, 1)

    def __new__(stop: int) -> range:
        return (0, stop, 1)

    def _get(self, idx: int) -> int:
        return self.start + (idx * self.step)

    def __getitem__(self, idx: int) -> int:
        n = self.__len__()
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError("range object index out of range")
        return self._get(idx)

    def __getitem__(self, s: Slice) -> range:
        if s.start is None and s.stop is None and s.step is None:
            return self
        else:
            start, stop, step, length = s.adjust_indices(self.__len__())
            substep = self.step * step
            substart = self._get(start)
            substop = self._get(stop)
            return range(substart, substop, substep)

    def __contains__(self, idx: int) -> bool:
        start, stop, step = self.start, self.stop, self.step
        if (step > 0 and not (start <= idx < stop)) or (
            step < 0 and not (stop < idx <= start)
        ):
            return False
        return (idx - start) % step == 0

    def _index(self, n: int) -> int:
        return (n - self.start) // self.step

    def index(self, n: int) -> int:
        if n in self:
            return self._index(n)
        else:
            raise ValueError(str(n) + " is not in range")

    def count(self, n: int) -> int:
        return int(n in self)

    def __iter__(self) -> Generator[int]:
        start, stop, step = self.start, self.stop, self.step
        i = start
        if step > 0:
            while i < stop:
                yield i
                i += step
        else:
            while i > stop:
                yield i
                i += step

    def __len__(self) -> int:
        start, stop, step = self.start, self.stop, self.step
        if step > 0 and start < stop:
            return 1 + (stop - 1 - start) // step
        elif step < 0 and start > stop:
            return 1 + (start - 1 - stop) // (-step)
        else:
            return 0

    def __bool__(self) -> bool:
        return self.__len__() > 0

    def __reversed__(self) -> Generator[int]:
        start, stop, step = self.start, self.stop, self.step
        n = self.__len__()
        return range(start + (n - 1) * step, start - step, -step).__iter__()

    def __repr__(self) -> str:
        if self.step == 1:
            return f"range({self.start}, {self.stop})"
        else:
            return f"range({self.start}, {self.stop}, {self.step})"

@overload
def staticrange(start: Static[int], stop: Static[int], step: Static[int] = 1):
    return range(start, stop, step)

@overload
def staticrange(stop: Static[int]):
    return range(0, stop, 1)
