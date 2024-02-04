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
class Optional:
    def __new__() -> Optional[T]:
        if isinstance(T, ByVal):
            return __internal__.opt_tuple_new(T)
        elif __has_rtti__(T):
            return __internal__.opt_ref_new_rtti(T)
        else:
            return __internal__.opt_ref_new(T)

    def __new__(what: T) -> Optional[T]:
        if isinstance(T, ByVal):
            return __internal__.opt_tuple_new_arg(what, T)
        elif __has_rtti__(T):
            return __internal__.opt_ref_new_arg_rtti(what, T)
        else:
            return __internal__.opt_ref_new_arg(what, T)

    def __has__(self) -> bool:
        if isinstance(T, ByVal):
            return __internal__.opt_tuple_bool(self, T)
        elif __has_rtti__(T):
            return __internal__.opt_ref_bool_rtti(T)
        else:
            return __internal__.opt_ref_bool(self, T)

    def __val__(self) -> T:
        if isinstance(T, ByVal):
            return __internal__.opt_tuple_invert(self, T)
        elif __has_rtti__(T):
            return __internal__.opt_ref_invert_rtti(T)
        else:
            return __internal__.opt_ref_invert(self, T)

    def __val_or__(self, default: T):
        if self.__has__():
            return self.__val__()
        return default

    def __bool__(self) -> bool:
        if not self.__has__():
            return False
        if hasattr(self.__val__(), "__bool__"):
            return self.__val__().__bool__()
        else:
            return True

    def __eq__(self, other: T) -> bool:
        if self is None:
            return False
        return self.__val__() == other

    def __eq__(self, other: Optional[T]) -> bool:
        if (self is None) or (other is None):
            return (self is None) and (other is None)
        return self.__val__() == other.__val__()

    def __ne__(self, other: T) -> bool:
        if self is None:
            return True
        return self.__val__() != other

    def __ne__(self, other: Optional[T]) -> bool:
        if (self is None) or (other is None):
            return not ((self is None) and (other is None))
        return self.__val__() != other.__val__()

    def __str__(self) -> str:
        return "None" if self is None else str(self.__val__())

    def __repr__(self) -> str:
        return "None" if self is None else self.__val__().__repr__()

    def __is_optional__(self, other: Optional[T]) -> bool:
        self_has = self.__has__()
        other_has = other.__has__()
        if (not self_has) or (not other_has):
            return (not self_has) and (not other_has)
        return self.__val__() is other.__val__()

optional = Optional

def unwrap(opt: Optional[T], T: type) -> T:
    if opt.__has__():
        return opt.__val__()
    raise ValueError("optional is None")
