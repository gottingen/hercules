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

SERVICE_DESCRIPTOR_MAGIC:int  = 0x14159bc3
MESSAGE_DESCRIPTOR_MAGIC:int  = 0x28aaeef9
ENUM_DESCRIPTOR_MAGIC:int =        0x114315af

class ProtobufFieldFlag:
    evalue: int

    FIELD_FLAG_PACKED: ClassVar[int] = 1<<0
    FIELD_FLAG_DEPRECATED: ClassVar[int] = 1<<1
    FIELD_FLAG_REQUIRED: ClassVar[int] = 1<<2

    def __init__(self) -> None:
        self.evalue = ProtobufFieldFlag.FIELD_FLAG_PACKED

    def __init__(self, ev: int) -> None:
        self.evalue = ev

    def __eq__(self, other: int) -> bool:
        return self.evalue == other

    def __eq__(self, other: ProtobufFieldFlag) -> bool:
        return self.evalue == other.evalue

    def __ne__(self, other: int) -> bool:
        return self.evalue != other

    def __ne__(self, other: ProtobufFieldFlag) -> bool:
        return self.evalue != other.evalue

    def __str__(self) -> str:
        return 'EnumTest: ' + str(self.evalue)

    def get_evalue(self) -> int:
        return self.evalue

    def set_evalue(self, v: int) -> None:
        self.evalue = v

class ProtobufLabel:
    evalue: int

    LABEL_OPTIONAL: ClassVar[int] = 0
    LABEL_REQUIRED: ClassVar[int] = 1
    LABEL_REPEATED: ClassVar[int] = 2
    LABEL_NONE: ClassVar[int] = 3

    def __init__(self) -> None:
        self.evalue = ProtobufLabel.LABEL_OPTIONAL

    def __init__(self, ev: int) -> None:
        self.evalue = ev

    def __eq__(self, other: int) -> bool:
        return self.evalue == other

    def __eq__(self, other: ProtobufLabel) -> bool:
        return self.evalue == other.evalue

    def __ne__(self, other: int) -> bool:
        return self.evalue != other

    def __ne__(self, other: ProtobufLabel) -> bool:
        return self.evalue != other.evalue

    def __str__(self) -> str:
        return 'EnumTest: ' + str(self.evalue)

    def get_evalue(self) -> int:
        return self.evalue

    def set_evalue(self, v: int) -> None:
        self.evalue = v


class ProtobufType:
    evalue: int

    TYPE_INT32: ClassVar[int] = 0
    TYPE_SINT32: ClassVar[int] = 1
    TYPE_SFIXED32: ClassVar[int] = 2
    TYPE_INT64: ClassVar[int] = 3
    TYPE_SINT64: ClassVar[int] = 4
    TYPE_SFIXED64: ClassVar[int] = 5
    TYPE_UINT32: ClassVar[int] = 6
    TYPE_FIXED32: ClassVar[int] = 7
    TYPE_UINT64: ClassVar[int] = 8
    TYPE_FIXED64: ClassVar[int] = 9
    TYPE_FLOAT: ClassVar[int] = 10
    TYPE_DOUBLE: ClassVar[int] = 11
    TYPE_BOOL: ClassVar[int] = 12
    TYPE_ENUM: ClassVar[int] = 13
    TYPE_STRING: ClassVar[int] = 14
    TYPE_BYTES: ClassVar[int] = 15
    TYPE_MESSAGE: ClassVar[int] = 16


    def __init__(self) -> None:
        self.evalue = ProtobufType.TYPE_DOUBLE

    def __init__(self, ev: int) -> None:
        self.evalue = ev

    def __eq__(self, other: int) -> bool:
        return self.evalue == other

    def __eq__(self, other: ProtobufType) -> bool:
        return self.evalue == other.evalue

    def __ne__(self, other: int) -> bool:
        return self.evalue != other

    def __ne__(self, other: ProtobufType) -> bool:
        return self.evalue != other.evalue

    def __str__(self) -> str:
        return 'EnumTest: ' + str(self.evalue)

    def get_evalue(self) -> int:
        return self.evalue

    def set_evalue(self, v: int) -> None:
        self.evalue = v

class ProtobufWireType:
    evalue: int

    WIRE_VARINT: ClassVar[int] = 0
    WIRE__64BIT: ClassVar[int] = 1
    WIRE_TYPE_LENGTH_PREFIXED: ClassVar[int] = 2
    WIRE_START_GROUP: ClassVar[int] = 3 # not supported
    WIRE_END_GROUP: ClassVar[int] = 4  # not supported
    WIRE_32BIT: ClassVar[int] = 5

    def __init__(self) -> None:
        self.evalue = ProtobufWireType.WIRE_VARINT

    def __init__(self, ev: int) -> None:
        self.evalue = ev

    def __eq__(self, other: int) -> bool:
        return self.evalue == other

    def __eq__(self, other: ProtobufWireType) -> bool:
        return self.evalue == other.evalue

    def __ne__(self, other: int) -> bool:
        return self.evalue != other

    def __ne__(self, other: ProtobufWireType) -> bool:
        return self.evalue != other.evalue

    def __str__(self) -> str:
        return 'EnumTest: ' + str(self.evalue)

    def get_evalue(self) -> int:
        return self.evalue

    def set_evalue(self, v: int) -> None:
        self.evalue = v


class ProtobufEnumValue:
    name : str
    value : int

    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value

    def __str__(self) -> str:
        return f"{self.name} = {self.value}"

    def get_name(self) -> str:
        return self.name

    def get_value(self) -> int:
        return self.value

    def set_name(self, name: str) -> None:
        self.name = name

    def set_value(self, value: int) -> None:
        self.value = value



class ProtobufEnumValue:
    name : str
    value : int

    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value

    def __str__(self) -> str:
        return f"{self.name} = {self.value}"

    def get_name(self) -> str:
        return self.name

    def get_value(self) -> int:
        return self.value

    def set_name(self, name: str) -> None:
        self.name = name

    def set_value(self, value: int) -> None:
        self.value = value

class ProtobufEnumValueIndex:
    name : str
    index : int

    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index

    def __str__(self) -> str:
        return f"{self.name} = {self.index}"

    def get_name(self) -> str:
        return self.name

    def get_index(self) -> int:
        return self.index

    def set_name(self, name: str) -> None:
        self.name = name

    def set_index(self, index: int) -> None:
        self.index = index

class ProtobufEnumDescriptor:
    magic: int
    name: str
    short_name: str
    package_name: str
    values: List[ProtobufEnumValue]
    values_by_name: List[ProtobufEnumValueIndex]
    values_ranges: List[IntRange]
    reserved1: cobj
    reserved2: cobj
    reserved3: cobj
    reserved4: cobj

    def __init__(self) -> None:
        self.magic = ENUM_DESCRIPTOR_MAGIC
        self.name = ""
        self.short_name = ""
        self.package_name = ""
        self.values = []
        self.values_by_name = []
        self.values_ranges = []
        self.reserved1 = cobj()
        self.reserved2 = cobj()
        self.reserved3 = cobj()
        self.reserved4 = cobj()

    def __str__(self) -> str:
        return f"EnumDescriptor: {self.name} ({self.short_name})"

    def get_magic(self) -> int:
        return self.magic

    def get_name(self) -> str:
        return self.name

    def get_short_name(self) -> str:
        return self.short_name

    def get_package_name(self) -> str:
        return self.package_name

    def get_values(self) -> List[ProtobufEnumValue]:
        return self.values

    def get_values_by_name(self) -> List[ProtobufEnumValueIndex]:
        return self.values_by_name

    def get_values_ranges(self) -> List[IntRange]:
        return self.values_ranges

