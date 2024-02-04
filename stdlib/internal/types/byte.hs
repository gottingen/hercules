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

@extend
class byte:
    @pure
    @llvm
    def __new__() -> byte:
        ret i8 0

    def __new__(b: byte) -> byte:
        return b

    def __new__(s: str) -> byte:
        if s.__len__() != 1:
            raise ValueError("str length must be 1 in byte constructor")
        return s.ptr[0]

    @pure
    @llvm
    def __new__(i: int) -> byte:
        %0 = trunc i64 %i to i8
        ret i8 %0

    def __copy__(self) -> byte:
        return self

    def __deepcopy__(self) -> byte:
        return self

    @pure
    @llvm
    def __bool__(self) -> bool:
        %0 = icmp ne i8 %self, 0
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __eq__(self, other: byte) -> bool:
        %0 = icmp eq i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    def __eq__(self, other: int) -> bool:
        return self == byte(other)

    @pure
    @llvm
    def __ne__(self, other: byte) -> bool:
        %0 = icmp ne i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __lt__(self, other: byte) -> bool:
        %0 = icmp ult i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __gt__(self, other: byte) -> bool:
        %0 = icmp ugt i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __le__(self, other: byte) -> bool:
        %0 = icmp ule i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __ge__(self, other: byte) -> bool:
        %0 = icmp uge i8 %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    def __str__(self) -> str:
        p = Ptr[byte](1)
        p[0] = self
        return str(p, 1)

    def __repr__(self) -> str:
        return f"byte({str(__ptr__(self), 1).__repr__()})"

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
