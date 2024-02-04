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

# Warning(!): This type must be consistent with the exception
# header type defined in runtime/exc.cpp.
class BaseException:
    _pytype: ClassVar[cobj] = cobj()
    typename: str
    message: str
    func: str
    file: str
    line: int
    col: int
    python_type: cobj

    def __init__(self, typename: str, message: str = ""):
        self.typename = typename
        self.message = message
        self.func = ""
        self.file = ""
        self.line = 0
        self.col = 0
        self.python_type = BaseException._pytype

    def __str__(self):
        return self.message

    def __repr__(self):
        return f'{self.typename}({self.message.__repr__()})'

class Exception(Static[BaseException]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, typename: str, msg: str = ""):
        super().__init__(typename, msg)
        if (hasattr(self.__class__, "_pytype")):
            self.python_type = self.__class__._pytype

class NameError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("NameError", message)
        self.python_type = self.__class__._pytype

class OSError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("OSError", message)
        self.python_type = self.__class__._pytype

class IOError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("IOError", message)
        self.python_type = self.__class__._pytype

class ValueError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("ValueError", message)
        self.python_type = self.__class__._pytype

class LookupError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, typename: str, message: str = ""):
        super().__init__(typename, message)
        self.python_type = self.__class__._pytype
    def __init__(self, msg: str = ""):
        super().__init__("LookupError", msg)
        self.python_type = self.__class__._pytype

class IndexError(Static[LookupError]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("IndexError", message)
        self.python_type = self.__class__._pytype

class KeyError(Static[LookupError]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("KeyError", message)
        self.python_type = self.__class__._pytype

class CError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("CError", message)
        self.python_type = self.__class__._pytype

class PyError(Static[Exception]):
    pytype: pyobj

    def __init__(self, message: str, pytype: pyobj = pyobj(cobj(), steal=True)):
        super().__init__("PyError", message)
        self.pytype = pytype

class TypeError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("TypeError", message)
        self.python_type = self.__class__._pytype

class ArithmeticError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, msg: str = ""):
        super().__init__("ArithmeticError", msg)
        self.python_type = self.__class__._pytype

class ZeroDivisionError(Static[ArithmeticError]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, typename: str, message: str = ""):
        super().__init__(typename, message)
        self.python_type = self.__class__._pytype
    def __init__(self, message: str = ""):
        super().__init__("ZeroDivisionError", message)
        self.python_type = self.__class__._pytype

class OverflowError(Static[ArithmeticError]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("OverflowError", message)
        self.python_type = self.__class__._pytype

class AttributeError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("AttributeError", message)
        self.python_type = self.__class__._pytype

class RuntimeError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, typename: str, message: str = ""):
        super().__init__(typename, message)
        self.python_type = self.__class__._pytype
    def __init__(self, message: str = ""):
        super().__init__("RuntimeError", message)
        self.python_type = self.__class__._pytype

class NotImplementedError(Static[RuntimeError]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("NotImplementedError", message)
        self.python_type = self.__class__._pytype

class StopIteration(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("StopIteration", message)
        self.python_type = self.__class__._pytype

class AssertionError(Static[Exception]):
    _pytype: ClassVar[cobj] = cobj()
    def __init__(self, message: str = ""):
        super().__init__("AssertionError", message)
        self.python_type = self.__class__._pytype

class SystemExit(Static[BaseException]):
    _pytype: ClassVar[cobj] = cobj()
    _status: int

    def __init__(self, message: str = "", status: int = 0):
        super().__init__("SystemExit", message)
        self._status = status
        self.python_type = self.__class__._pytype

    def __init__(self, status: int):
        self.__init__("", status)

    @property
    def status(self):
        return self._status

class StaticCompileError(Static[Exception]):
    def __init__(self, message: str = ""):
        super().__init__("StaticCompileError", message)
