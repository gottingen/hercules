#
# Copyright 2024 EA Authors.
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

class IntRange:
    start_value: int
    orig_index: int

    def __init__(self, start: int, index: int) -> None:
        self.start_value = start
        self.orig_index = index

    def __str__(self) -> str:
        return 'IntRange: ' + str(self.start_value) + ', ' + str(self.orig_index)

    def get_start_value(self) -> int:
        return self.start_value

    def set_start_value(self, v: int) -> None:
        self.start_value = v

    def get_orig_index(self) -> int:
        return self.orig_index

    def set_orig_index(self, v: int) -> None:
        self.orig_index = v

    def __eq__(self, other: Ptr[IntRange]) -> bool:
        return self.start_value == other.start_value and self.orig_index == other.orig_index

    def __ne__(self, other: Ptr[IntRange]) -> bool:
        return self.start_value != other.start_value or self.orig_index != other.orig_index

