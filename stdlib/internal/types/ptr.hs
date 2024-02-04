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
class Ptr:
    @pure
    @llvm
    def __new__() -> Ptr[T]:
        ret ptr null

    @__internal__
    def __new__(sz: int) -> Ptr[T]:
        pass

    @pure
    @derives
    @llvm
    def __new__(other: Ptr[T]) -> Ptr[T]:
        ret ptr %other

    @pure
    @derives
    @llvm
    def __new__(other: Ptr[byte]) -> Ptr[T]:
        ret ptr %other

    @pure
    @llvm
    def __int__(self) -> int:
        %0 = ptrtoint ptr %self to i64
        ret i64 %0

    @pure
    @llvm
    def __copy__(self) -> Ptr[T]:
        ret ptr %self

    @pure
    @llvm
    def __bool__(self) -> bool:
        %0 = icmp ne ptr %self, null
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @derives
    @llvm
    def __getitem__(self, index: int) -> T:
        %0 = getelementptr {=T}, ptr %self, i64 %index
        %1 = load {=T}, ptr %0
        ret {=T} %1

    @self_captures
    @llvm
    def __setitem__(self, index: int, what: T) -> None:
        %0 = getelementptr {=T}, ptr %self, i64 %index
        store {=T} %what, ptr %0
        ret {} {}

    @pure
    @derives
    @llvm
    def __add__(self, other: int) -> Ptr[T]:
        %0 = getelementptr {=T}, ptr %self, i64 %other
        ret ptr %0

    @pure
    @llvm
    def __sub__(self, other: Ptr[T]) -> int:
        %0 = ptrtoint ptr %self to i64
        %1 = ptrtoint ptr %other to i64
        %2 = sub i64 %0, %1
        %3 = sdiv exact i64 %2, ptrtoint (ptr getelementptr ({=T}, {=T}* null, i32 1) to i64)
        ret i64 %3

    @pure
    @derives
    @llvm
    def __sub__(self, other: int) -> Ptr[T]:
        %0 = sub i64 0, %other
        %1 = getelementptr {=T}, {=T}* %self, i64 %0
        ret {=T}* %1

    @pure
    @llvm
    def __eq__(self, other: Ptr[T]) -> bool:
        %0 = icmp eq ptr %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __ne__(self, other: Ptr[T]) -> bool:
        %0 = icmp ne ptr %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __lt__(self, other: Ptr[T]) -> bool:
        %0 = icmp slt ptr %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __gt__(self, other: Ptr[T]) -> bool:
        %0 = icmp sgt ptr %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __le__(self, other: Ptr[T]) -> bool:
        %0 = icmp sle ptr %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @pure
    @llvm
    def __ge__(self, other: Ptr[T]) -> bool:
        %0 = icmp sge ptr %self, %other
        %1 = zext i1 %0 to i8
        ret i8 %1

    @nocapture
    @llvm
    def __prefetch_r0__(self) -> None:
        declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32)
        call void @llvm.prefetch(ptr %self, i32 0, i32 0, i32 1)
        ret {} {}

    @nocapture
    @llvm
    def __prefetch_r1__(self) -> None:
        declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32)
        call void @llvm.prefetch(ptr %self, i32 0, i32 1, i32 1)
        ret {} {}

    @nocapture
    @llvm
    def __prefetch_r2__(self) -> None:
        declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32)
        call void @llvm.prefetch(ptr %self, i32 0, i32 2, i32 1)
        ret {} {}

    @nocapture
    @llvm
    def __prefetch_r3__(self) -> None:
        declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32)
        call void @llvm.prefetch(ptr %self, i32 0, i32 3, i32 1)
        ret {} {}

    @nocapture
    @llvm
    def __prefetch_w0__(self) -> None:
        declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32)
        call void @llvm.prefetch(ptr %self, i32 1, i32 0, i32 1)
        ret {} {}

    @nocapture
    @llvm
    def __prefetch_w1__(self) -> None:
        declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32)
        call void @llvm.prefetch(ptr %self, i32 1, i32 1, i32 1)
        ret {} {}

    @nocapture
    @llvm
    def __prefetch_w2__(self) -> None:
        declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32)
        call void @llvm.prefetch(ptr %self, i32 1, i32 2, i32 1)
        ret {} {}

    @nocapture
    @llvm
    def __prefetch_w3__(self) -> None:
        declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32)
        call void @llvm.prefetch(ptr %self, i32 1, i32 3, i32 1)
        ret {} {}

    @pure
    @derives
    @llvm
    def as_byte(self) -> Ptr[byte]:
        ret ptr %self

    def __repr__(self) -> str:
        return self.__format__("")

ptr = Ptr
Jar = Ptr[byte]

# Forward declarations
class List:
    len: int
    arr: Array[T]
    T: type

@extend
class NoneType:
    def __new__() -> NoneType:
        return ()

    def __eq__(self, other: NoneType):
        return True

    def __ne__(self, other: NoneType):
        return False

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "None"
