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
class Slice:
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]

    def __new__(stop: Optional[int]):
        return Slice(None, stop, None)

    def adjust_indices(self, length: int) -> Tuple[int, int, int, int]:
        step: int = self.step if self.step is not None else 1
        start: int = 0
        stop: int = 0
        if step == 0:
            raise ValueError("slice step cannot be zero")
        if step > 0:
            start = self.start if self.start is not None else 0
            stop = self.stop if self.stop is not None else length
        else:
            start = self.start if self.start is not None else length - 1
            stop = self.stop if self.stop is not None else -(length + 1)

        return Slice.adjust_indices_helper(length, start, stop, step)

    def adjust_indices_helper(
        length: int, start: int, stop: int, step: int
    ) -> Tuple[int, int, int, int]:
        if start < 0:
            start += length
            if start < 0:
                start = -1 if step < 0 else 0
        elif start >= length:
            start = length - 1 if step < 0 else length

        if stop < 0:
            stop += length
            if stop < 0:
                stop = -1 if step < 0 else 0
        elif stop >= length:
            stop = length - 1 if step < 0 else length

        if step < 0:
            if stop < start:
                return start, stop, step, (start - stop - 1) // (-step) + 1
        else:
            if start < stop:
                return start, stop, step, (stop - start - 1) // step + 1

        return start, stop, step, 0

    def indices(self, length: int):
        if length < 0:
            raise ValueError("length should not be negative")
        return self.adjust_indices(length)[:-1]

    def __repr__(self):
        return f"slice({self.start}, {self.stop}, {self.step})"

slice = Slice
