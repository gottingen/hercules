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

from .int_range import IntRange
from .enums import *

class ProtobufFieldDescriptor:
    name: str
    id: int
    label: ProtobufLabel
    type: ProtobufType
    quantifier_offset: int
    offset: int
    descriptor:Ptr[ProtobufMessageDescriptor]
    defualt_value: cobj

    def __init__(self) -> None:
        self.name = ''
        self.id = 0
        self.label = ProtobufLabel.LABEL_OPTIONAL
        self.type = ProtobufType.TYPE_INT32
        self.quantifier_offset = 0
        self.offset = 0
        self.descriptor = None
        self.default_value = None

    def __str__(self) -> str:
        return 'FieldDescriptor: ' + self.name + ', ' + str(self.id) + ', ' + str(self.label) + ', ' + str(self.type) + ', ' + str(self.quantifier_offset) + ', ' + str(self.offset) + ', ' + str(self.descriptor) + ', ' + str(self.default_value)

    def get_name(self) -> str:
        return self.name

    def set_name(self, v: str) -> None:
        self.name = v

    def get_id(self) -> int:
        return self.id

    def set_id(self, v: int) -> None:
        self.id = v

    def get_label(self) -> ProtobufLabel:
        return self.label

    def set_label(self, v: ProtobufLabel) -> None:
        self.label = v

    def get_type(self) -> ProtobufType:
        return self.type

    def set_type(self, v: ProtobufType) -> None:
        self.type = v

    def get_quantifier_offset(self) -> int:
        return self.quantifier_offset

    def set_quantifier_offset(self, v: int) -> None:
        self.quantifier_offset = v

    def get_offset(self) -> int:
        return self.offset

    def set_offset(self, v: int) -> None:
        self.offset = v

    def get_descriptor(self) -> Ptr[ProtobufMessageDescriptor]:
        return self.descriptor

    def set_descriptor(self, v: Ptr[ProtobufMessageDescriptor]) -> None:
        self.descriptor = v

    def get_default_value(self) -> cobj:
        return self.default_value

    def set_default_value(self, v: cobj) -> None:
        self.default_value = v

class ProtobufMessageDescriptor:
    magic: int
    name: str
    short_name: str
    package_name: str
    fields: List[ProtobufFieldDescriptor]
    fields_sorted_by_name: List[int]