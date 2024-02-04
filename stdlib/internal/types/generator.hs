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
class Generator:
    @__internal__
    def __promise__(self) -> Ptr[T]:
        pass

    def done(self) -> bool:
        self.__resume__()
        return self.__done__()

    def next(self: Generator[T]) -> T:
        if isinstance(T, None):
            pass
        else:
            return self.__promise__()[0]

    def __iter__(self) -> Generator[T]:
        return self

    @pure
    @llvm
    def __raw__(self) -> Ptr[byte]:
        ret ptr %self

    @pure
    @derives
    @llvm
    def __new__(ptr: cobj) -> Generator[T]:
        ret ptr %ptr

    @pure
    @llvm
    def __done__(self) -> bool:
        declare i1 @llvm.coro.done(ptr nocapture readonly)
        %0 = call i1 @llvm.coro.done(ptr %self)
        %1 = zext i1 %0 to i8
        ret i8 %1

    @nocapture
    @llvm
    def __resume__(self) -> None:
        declare void @llvm.coro.resume(ptr)
        call void @llvm.coro.resume(ptr %self)
        ret {} {}

    def __repr__(self) -> str:
        return __internal__.raw_type_str(self.__raw__(), "generator")

    def send(self, what: T) -> T:
        p = self.__promise__()
        p[0] = what
        self.__resume__()
        return p[0]

    @nocapture
    @llvm
    def destroy(self) -> None:
        declare void @llvm.coro.destroy(ptr)
        call void @llvm.coro.destroy(ptr %self)
        ret {} {}

generator = Generator
