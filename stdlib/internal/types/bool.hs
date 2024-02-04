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

from internal.attributes import commutative, associative

@extend
class bool:
    def __new__() -> bool:
        return False

    def __new__(what) -> bool:
        return what.__bool__()

    def __repr__(self) -> str:
        return "True" if self else "False"

    def __copy__(self) -> bool:
        return self

    def __deepcopy__(self) -> bool:
        return self

    def __bool__(self) -> bool:
        return self

    def __hash__(self) -> int:
        return int(self)

    @pure
    @llvm
    def __invert__(self) -> bool:
        %0 = trunc i8 %self to i1
        %1 = xor i1 %0, true
        %2 = zext i1 %1 to i8
        ret i8 %2

    @pure
    @llvm
    def __eq__(self, other: bool) -> bool:
        %0 = icmp eq i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __ne__(self, other: bool) -> bool:
        %0 = icmp ne i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __lt__(self, other: bool) -> bool:
        %0 = icmp ult i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __gt__(self, other: bool) -> bool:
        %0 = icmp ugt i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __le__(self, other: bool) -> bool:
        %0 = icmp ule i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __ge__(self, other: bool) -> bool:
        %0 = icmp uge i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @commutative
    @associative
    @llvm
    def __and__(self, other: bool) -> bool:
        %0 = and i8 %self, %other
        ret i8 %0

    @pure
    @commutative
    @associative
    @llvm
    def __or__(self, other: bool) -> bool:
        %0 = or i8 %self, %other
        ret i8 %0

    @pure
    @commutative
    @associative
    @llvm
    def __xor__(self, other: bool) -> bool:
        %0 = xor i8 %self, %other
        ret i8 %0

    @pure
    @llvm
    def __int__(self) -> int:
        %0 = zext i8 %self to i64
        ret i64 %0

    @pure
    @llvm
    def __float__(self) -> float:
        %0 = uitofp i8 %self to double
        ret double %0
