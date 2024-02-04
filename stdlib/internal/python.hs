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

import os

from gc import atomic, alloc_uncollectable
from internal.dlopen import *

# general
Py_DecRef = Function[[cobj], NoneType](cobj())
Py_IncRef = Function[[cobj], NoneType](cobj())
Py_Initialize = Function[[], NoneType](cobj())
PyImport_AddModule = Function[[cobj], cobj](cobj())
PyImport_AddModuleObject = Function[[cobj], cobj](cobj())
PyImport_ImportModule = Function[[cobj], cobj](cobj())
PyRun_SimpleString = Function[[cobj], NoneType](cobj())
PyEval_GetGlobals = Function[[], cobj](cobj())
PyEval_GetBuiltins = Function[[], cobj](cobj())

# conversions
PyLong_AsLong = Function[[cobj], int](cobj())
PyLong_FromLong = Function[[int], cobj](cobj())
PyFloat_AsDouble = Function[[cobj], float](cobj())
PyFloat_FromDouble = Function[[float], cobj](cobj())
PyBool_FromLong = Function[[int], cobj](cobj())
PyBytes_AsString = Function[[cobj], cobj](cobj())
PyList_New = Function[[int], cobj](cobj())
PyList_Size = Function[[cobj], int](cobj())
PyList_GetItem = Function[[cobj, int], cobj](cobj())
PyList_SetItem = Function[[cobj, int, cobj], cobj](cobj())
PyDict_New = Function[[], cobj](cobj())
PyDict_Next = Function[[cobj, Ptr[int], Ptr[cobj], Ptr[cobj]], int](cobj())
PyDict_GetItem = Function[[cobj, cobj], cobj](cobj())
PyDict_GetItemString = Function[[cobj, cobj], cobj](cobj())
PyDict_SetItem = Function[[cobj, cobj, cobj], cobj](cobj())
PyDict_Size = Function[[cobj], int](cobj())
PySet_Add = Function[[cobj, cobj], cobj](cobj())
PySet_New = Function[[cobj], cobj](cobj())
PyTuple_New = Function[[int], cobj](cobj())
PyTuple_Size = Function[[cobj], int](cobj())
PyTuple_GetItem = Function[[cobj, int], cobj](cobj())
PyTuple_SetItem = Function[[cobj, int, cobj], NoneType](cobj())
PyUnicode_AsEncodedString = Function[[cobj, cobj, cobj], cobj](cobj())
PyUnicode_DecodeFSDefaultAndSize = Function[[cobj, int], cobj](cobj())
PyUnicode_FromString = Function[[cobj], cobj](cobj())
PyComplex_FromDoubles = Function[[float, float], cobj](cobj())
PyComplex_RealAsDouble = Function[[cobj], float](cobj())
PyComplex_ImagAsDouble = Function[[cobj], float](cobj())
PyIter_Next = Function[[cobj], cobj](cobj())
PySlice_New = Function[[cobj, cobj, cobj], cobj](cobj())
PySlice_Unpack = Function[[cobj, Ptr[int], Ptr[int], Ptr[int]], int](cobj())
PyCapsule_New = Function[[cobj, cobj, cobj], cobj](cobj())
PyCapsule_GetPointer = Function[[cobj, cobj], cobj](cobj())

# number
PyNumber_Add = Function[[cobj, cobj], cobj](cobj())
PyNumber_Subtract = Function[[cobj, cobj], cobj](cobj())
PyNumber_Multiply = Function[[cobj, cobj], cobj](cobj())
PyNumber_MatrixMultiply = Function[[cobj, cobj], cobj](cobj())
PyNumber_FloorDivide = Function[[cobj, cobj], cobj](cobj())
PyNumber_TrueDivide = Function[[cobj, cobj], cobj](cobj())
PyNumber_Remainder = Function[[cobj, cobj], cobj](cobj())
PyNumber_Divmod = Function[[cobj, cobj], cobj](cobj())
PyNumber_Power = Function[[cobj, cobj, cobj], cobj](cobj())
PyNumber_Negative = Function[[cobj], cobj](cobj())
PyNumber_Positive = Function[[cobj], cobj](cobj())
PyNumber_Absolute = Function[[cobj], cobj](cobj())
PyNumber_Invert = Function[[cobj], cobj](cobj())
PyNumber_Lshift = Function[[cobj, cobj], cobj](cobj())
PyNumber_Rshift = Function[[cobj, cobj], cobj](cobj())
PyNumber_And = Function[[cobj, cobj], cobj](cobj())
PyNumber_Xor = Function[[cobj, cobj], cobj](cobj())
PyNumber_Or = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceAdd = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceSubtract = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceMultiply = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceMatrixMultiply = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceFloorDivide = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceTrueDivide = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceRemainder = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlacePower = Function[[cobj, cobj, cobj], cobj](cobj())
PyNumber_InPlaceLshift = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceRshift = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceAnd = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceXor = Function[[cobj, cobj], cobj](cobj())
PyNumber_InPlaceOr = Function[[cobj, cobj], cobj](cobj())
PyNumber_Long = Function[[cobj], cobj](cobj())
PyNumber_Float = Function[[cobj], cobj](cobj())
PyNumber_Index = Function[[cobj], cobj](cobj())

# object
PyObject_Call = Function[[cobj, cobj, cobj], cobj](cobj())
PyObject_GetAttr = Function[[cobj, cobj], cobj](cobj())
PyObject_GetAttrString = Function[[cobj, cobj], cobj](cobj())
PyObject_GetIter = Function[[cobj], cobj](cobj())
PyObject_HasAttrString = Function[[cobj, cobj], int](cobj())
PyObject_IsTrue = Function[[cobj], int](cobj())
PyObject_Length = Function[[cobj], int](cobj())
PyObject_LengthHint = Function[[cobj, int], int](cobj())
PyObject_SetAttrString = Function[[cobj, cobj, cobj], cobj](cobj())
PyObject_Str = Function[[cobj], cobj](cobj())
PyObject_Repr = Function[[cobj], cobj](cobj())
PyObject_Hash = Function[[cobj], int](cobj())
PyObject_GetItem = Function[[cobj, cobj], cobj](cobj())
PyObject_SetItem = Function[[cobj, cobj, cobj], int](cobj())
PyObject_DelItem = Function[[cobj, cobj], int](cobj())
PyObject_RichCompare = Function[[cobj, cobj, i32], cobj](cobj())
PyObject_IsInstance = Function[[cobj, cobj], i32](cobj())

# error handling
PyErr_Fetch = Function[[Ptr[cobj], Ptr[cobj], Ptr[cobj]], NoneType](cobj())
PyErr_NormalizeException = Function[[Ptr[cobj], Ptr[cobj], Ptr[cobj]], NoneType](cobj())
PyErr_SetString = Function[[cobj, cobj], NoneType](cobj())

# constants
Py_None = cobj()
Py_True = cobj()
Py_False = cobj()
Py_Ellipsis = cobj()
Py_NotImplemented = cobj()
Py_LT = 0
Py_LE = 1
Py_EQ = 2
Py_NE = 3
Py_GT = 4
Py_GE = 5

# types
PyLong_Type = cobj()
PyFloat_Type = cobj()
PyBool_Type = cobj()
PyUnicode_Type = cobj()
PyComplex_Type = cobj()
PyList_Type = cobj()
PyDict_Type = cobj()
PySet_Type = cobj()
PyTuple_Type = cobj()
PySlice_Type = cobj()
PyCapsule_Type = cobj()

# exceptions
PyExc_BaseException = cobj()
PyExc_Exception = cobj()
PyExc_NameError = cobj()
PyExc_OSError = cobj()
PyExc_IOError = cobj()
PyExc_ValueError = cobj()
PyExc_LookupError = cobj()
PyExc_IndexError = cobj()
PyExc_KeyError = cobj()
PyExc_TypeError = cobj()
PyExc_ArithmeticError = cobj()
PyExc_ZeroDivisionError = cobj()
PyExc_OverflowError = cobj()
PyExc_AttributeError = cobj()
PyExc_RuntimeError = cobj()
PyExc_NotImplementedError = cobj()
PyExc_StopIteration = cobj()
PyExc_AssertionError = cobj()
PyExc_SystemExit = cobj()

_PY_MODULE_CACHE = Dict[str, pyobj]()

_PY_INIT = """
import io

clsf = None
clsa = None
plt = None
try:
    import matplotlib.figure
    import matplotlib.pyplot
    plt = matplotlib.pyplot
    clsf = matplotlib.figure.Figure
    clsa = matplotlib.artist.Artist
except ModuleNotFoundError:
    pass

def __hercules_repr__(fig):
    if clsf and isinstance(fig, clsf):
        stream = io.StringIO()
        fig.savefig(stream, format="svg")
        return 'image/svg+xml', stream.getvalue()
    elif clsa and isinstance(fig, list) and all(
        isinstance(i, clsa) for i in fig
    ):
        stream = io.StringIO()
        plt.gcf().savefig(stream, format="svg")
        return 'image/svg+xml', stream.getvalue()
    elif hasattr(fig, "_repr_html_"):
        return 'text/html', fig._repr_html_()
    else:
        return 'text/plain', fig.__repr__()
"""

_PY_INITIALIZED = False

def init_handles_dlopen(py_handle: cobj):
    global Py_DecRef
    global Py_IncRef
    global Py_Initialize
    global PyImport_AddModule
    global PyImport_AddModuleObject
    global PyImport_ImportModule
    global PyRun_SimpleString
    global PyEval_GetGlobals
    global PyEval_GetBuiltins
    global PyLong_AsLong
    global PyLong_FromLong
    global PyFloat_AsDouble
    global PyFloat_FromDouble
    global PyBool_FromLong
    global PyBytes_AsString
    global PyList_New
    global PyList_Size
    global PyList_GetItem
    global PyList_SetItem
    global PyDict_New
    global PyDict_Next
    global PyDict_GetItem
    global PyDict_GetItemString
    global PyDict_SetItem
    global PyDict_Size
    global PySet_Add
    global PySet_New
    global PyTuple_New
    global PyTuple_Size
    global PyTuple_GetItem
    global PyTuple_SetItem
    global PyUnicode_AsEncodedString
    global PyUnicode_DecodeFSDefaultAndSize
    global PyUnicode_FromString
    global PyComplex_FromDoubles
    global PyComplex_RealAsDouble
    global PyComplex_ImagAsDouble
    global PyIter_Next
    global PySlice_New
    global PySlice_Unpack
    global PyCapsule_New
    global PyCapsule_GetPointer
    global PyNumber_Add
    global PyNumber_Subtract
    global PyNumber_Multiply
    global PyNumber_MatrixMultiply
    global PyNumber_FloorDivide
    global PyNumber_TrueDivide
    global PyNumber_Remainder
    global PyNumber_Divmod
    global PyNumber_Power
    global PyNumber_Negative
    global PyNumber_Positive
    global PyNumber_Absolute
    global PyNumber_Invert
    global PyNumber_Lshift
    global PyNumber_Rshift
    global PyNumber_And
    global PyNumber_Xor
    global PyNumber_Or
    global PyNumber_InPlaceAdd
    global PyNumber_InPlaceSubtract
    global PyNumber_InPlaceMultiply
    global PyNumber_InPlaceMatrixMultiply
    global PyNumber_InPlaceFloorDivide
    global PyNumber_InPlaceTrueDivide
    global PyNumber_InPlaceRemainder
    global PyNumber_InPlacePower
    global PyNumber_InPlaceLshift
    global PyNumber_InPlaceRshift
    global PyNumber_InPlaceAnd
    global PyNumber_InPlaceXor
    global PyNumber_InPlaceOr
    global PyNumber_Long
    global PyNumber_Float
    global PyNumber_Index
    global PyObject_Call
    global PyObject_GetAttr
    global PyObject_GetAttrString
    global PyObject_GetIter
    global PyObject_HasAttrString
    global PyObject_IsTrue
    global PyObject_Length
    global PyObject_LengthHint
    global PyObject_SetAttrString
    global PyObject_Str
    global PyObject_Repr
    global PyObject_Hash
    global PyObject_GetItem
    global PyObject_SetItem
    global PyObject_DelItem
    global PyObject_RichCompare
    global PyObject_IsInstance
    global PyErr_Fetch
    global PyErr_NormalizeException
    global PyErr_SetString
    global Py_None
    global Py_True
    global Py_False
    global Py_Ellipsis
    global Py_NotImplemented
    global PyLong_Type
    global PyFloat_Type
    global PyBool_Type
    global PyUnicode_Type
    global PyComplex_Type
    global PyList_Type
    global PyDict_Type
    global PySet_Type
    global PyTuple_Type
    global PySlice_Type
    global PyCapsule_Type
    global PyExc_BaseException
    global PyExc_Exception
    global PyExc_NameError
    global PyExc_OSError
    global PyExc_IOError
    global PyExc_ValueError
    global PyExc_LookupError
    global PyExc_IndexError
    global PyExc_KeyError
    global PyExc_TypeError
    global PyExc_ArithmeticError
    global PyExc_ZeroDivisionError
    global PyExc_OverflowError
    global PyExc_AttributeError
    global PyExc_RuntimeError
    global PyExc_NotImplementedError
    global PyExc_StopIteration
    global PyExc_AssertionError
    global PyExc_SystemExit

    Py_DecRef = dlsym(py_handle, "Py_DecRef")
    Py_IncRef = dlsym(py_handle, "Py_IncRef")
    Py_Initialize = dlsym(py_handle, "Py_Initialize")
    PyImport_AddModule = dlsym(py_handle, "PyImport_AddModule")
    PyImport_AddModuleObject = dlsym(py_handle, "PyImport_AddModuleObject")
    PyImport_ImportModule = dlsym(py_handle, "PyImport_ImportModule")
    PyRun_SimpleString = dlsym(py_handle, "PyRun_SimpleString")
    PyEval_GetGlobals = dlsym(py_handle, "PyEval_GetGlobals")
    PyEval_GetBuiltins = dlsym(py_handle, "PyEval_GetBuiltins")
    PyLong_AsLong = dlsym(py_handle, "PyLong_AsLong")
    PyLong_FromLong = dlsym(py_handle, "PyLong_FromLong")
    PyFloat_AsDouble = dlsym(py_handle, "PyFloat_AsDouble")
    PyFloat_FromDouble = dlsym(py_handle, "PyFloat_FromDouble")
    PyBool_FromLong = dlsym(py_handle, "PyBool_FromLong")
    PyBytes_AsString = dlsym(py_handle, "PyBytes_AsString")
    PyList_New = dlsym(py_handle, "PyList_New")
    PyList_Size = dlsym(py_handle, "PyList_Size")
    PyList_GetItem = dlsym(py_handle, "PyList_GetItem")
    PyList_SetItem = dlsym(py_handle, "PyList_SetItem")
    PyDict_New = dlsym(py_handle, "PyDict_New")
    PyDict_Next = dlsym(py_handle, "PyDict_Next")
    PyDict_GetItem = dlsym(py_handle, "PyDict_GetItem")
    PyDict_GetItemString = dlsym(py_handle, "PyDict_GetItemString")
    PyDict_SetItem = dlsym(py_handle, "PyDict_SetItem")
    PyDict_Size = dlsym(py_handle, "PyDict_Size")
    PySet_Add = dlsym(py_handle, "PySet_Add")
    PySet_New = dlsym(py_handle, "PySet_New")
    PyTuple_New = dlsym(py_handle, "PyTuple_New")
    PyTuple_Size = dlsym(py_handle, "PyTuple_Size")
    PyTuple_GetItem = dlsym(py_handle, "PyTuple_GetItem")
    PyTuple_SetItem = dlsym(py_handle, "PyTuple_SetItem")
    PyUnicode_AsEncodedString = dlsym(py_handle, "PyUnicode_AsEncodedString")
    PyUnicode_DecodeFSDefaultAndSize = dlsym(py_handle, "PyUnicode_DecodeFSDefaultAndSize")
    PyUnicode_FromString = dlsym(py_handle, "PyUnicode_FromString")
    PyComplex_FromDoubles = dlsym(py_handle, "PyComplex_FromDoubles")
    PyComplex_RealAsDouble = dlsym(py_handle, "PyComplex_RealAsDouble")
    PyComplex_ImagAsDouble = dlsym(py_handle, "PyComplex_ImagAsDouble")
    PyIter_Next = dlsym(py_handle, "PyIter_Next")
    PySlice_New = dlsym(py_handle, "PySlice_New")
    PySlice_Unpack = dlsym(py_handle, "PySlice_Unpack")
    PyCapsule_New = dlsym(py_handle, "PyCapsule_New")
    PyCapsule_GetPointer = dlsym(py_handle, "PyCapsule_GetPointer")
    PyNumber_Add = dlsym(py_handle, "PyNumber_Add")
    PyNumber_Subtract = dlsym(py_handle, "PyNumber_Subtract")
    PyNumber_Multiply = dlsym(py_handle, "PyNumber_Multiply")
    PyNumber_MatrixMultiply = dlsym(py_handle, "PyNumber_MatrixMultiply")
    PyNumber_FloorDivide = dlsym(py_handle, "PyNumber_FloorDivide")
    PyNumber_TrueDivide = dlsym(py_handle, "PyNumber_TrueDivide")
    PyNumber_Remainder = dlsym(py_handle, "PyNumber_Remainder")
    PyNumber_Divmod = dlsym(py_handle, "PyNumber_Divmod")
    PyNumber_Power = dlsym(py_handle, "PyNumber_Power")
    PyNumber_Negative = dlsym(py_handle, "PyNumber_Negative")
    PyNumber_Positive = dlsym(py_handle, "PyNumber_Positive")
    PyNumber_Absolute = dlsym(py_handle, "PyNumber_Absolute")
    PyNumber_Invert = dlsym(py_handle, "PyNumber_Invert")
    PyNumber_Lshift = dlsym(py_handle, "PyNumber_Lshift")
    PyNumber_Rshift = dlsym(py_handle, "PyNumber_Rshift")
    PyNumber_And = dlsym(py_handle, "PyNumber_And")
    PyNumber_Xor = dlsym(py_handle, "PyNumber_Xor")
    PyNumber_Or = dlsym(py_handle, "PyNumber_Or")
    PyNumber_InPlaceAdd = dlsym(py_handle, "PyNumber_InPlaceAdd")
    PyNumber_InPlaceSubtract = dlsym(py_handle, "PyNumber_InPlaceSubtract")
    PyNumber_InPlaceMultiply = dlsym(py_handle, "PyNumber_InPlaceMultiply")
    PyNumber_InPlaceMatrixMultiply = dlsym(py_handle, "PyNumber_InPlaceMatrixMultiply")
    PyNumber_InPlaceFloorDivide = dlsym(py_handle, "PyNumber_InPlaceFloorDivide")
    PyNumber_InPlaceTrueDivide = dlsym(py_handle, "PyNumber_InPlaceTrueDivide")
    PyNumber_InPlaceRemainder = dlsym(py_handle, "PyNumber_InPlaceRemainder")
    PyNumber_InPlacePower = dlsym(py_handle, "PyNumber_InPlacePower")
    PyNumber_InPlaceLshift = dlsym(py_handle, "PyNumber_InPlaceLshift")
    PyNumber_InPlaceRshift = dlsym(py_handle, "PyNumber_InPlaceRshift")
    PyNumber_InPlaceAnd = dlsym(py_handle, "PyNumber_InPlaceAnd")
    PyNumber_InPlaceXor = dlsym(py_handle, "PyNumber_InPlaceXor")
    PyNumber_InPlaceOr = dlsym(py_handle, "PyNumber_InPlaceOr")
    PyNumber_Long = dlsym(py_handle, "PyNumber_Long")
    PyNumber_Float = dlsym(py_handle, "PyNumber_Float")
    PyNumber_Index = dlsym(py_handle, "PyNumber_Index")
    PyObject_Call = dlsym(py_handle, "PyObject_Call")
    PyObject_GetAttr = dlsym(py_handle, "PyObject_GetAttr")
    PyObject_GetAttrString = dlsym(py_handle, "PyObject_GetAttrString")
    PyObject_GetIter = dlsym(py_handle, "PyObject_GetIter")
    PyObject_HasAttrString = dlsym(py_handle, "PyObject_HasAttrString")
    PyObject_IsTrue = dlsym(py_handle, "PyObject_IsTrue")
    PyObject_Length = dlsym(py_handle, "PyObject_Length")
    PyObject_LengthHint = dlsym(py_handle, "PyObject_LengthHint")
    PyObject_SetAttrString = dlsym(py_handle, "PyObject_SetAttrString")
    PyObject_Str = dlsym(py_handle, "PyObject_Str")
    PyObject_Repr = dlsym(py_handle, "PyObject_Repr")
    PyObject_Hash = dlsym(py_handle, "PyObject_Hash")
    PyObject_GetItem = dlsym(py_handle, "PyObject_GetItem")
    PyObject_SetItem = dlsym(py_handle, "PyObject_SetItem")
    PyObject_DelItem = dlsym(py_handle, "PyObject_DelItem")
    PyObject_RichCompare = dlsym(py_handle, "PyObject_RichCompare")
    PyObject_IsInstance = dlsym(py_handle, "PyObject_IsInstance")
    PyErr_Fetch = dlsym(py_handle, "PyErr_Fetch")
    PyErr_NormalizeException = dlsym(py_handle, "PyErr_NormalizeException")
    PyErr_SetString = dlsym(py_handle, "PyErr_SetString")
    Py_None = dlsym(py_handle, "_Py_NoneStruct")
    Py_True = dlsym(py_handle, "_Py_TrueStruct")
    Py_False = dlsym(py_handle, "_Py_FalseStruct")
    Py_Ellipsis = dlsym(py_handle, "_Py_EllipsisObject")
    Py_NotImplemented = dlsym(py_handle, "_Py_NotImplementedStruct")
    PyLong_Type = dlsym(py_handle, "PyLong_Type")
    PyFloat_Type = dlsym(py_handle, "PyFloat_Type")
    PyBool_Type = dlsym(py_handle, "PyBool_Type")
    PyUnicode_Type = dlsym(py_handle, "PyUnicode_Type")
    PyComplex_Type = dlsym(py_handle, "PyComplex_Type")
    PyList_Type = dlsym(py_handle, "PyList_Type")
    PyDict_Type = dlsym(py_handle, "PyDict_Type")
    PySet_Type = dlsym(py_handle, "PySet_Type")
    PyTuple_Type = dlsym(py_handle, "PyTuple_Type")
    PySlice_Type = dlsym(py_handle, "PySlice_Type")
    PyCapsule_Type = dlsym(py_handle, "PyCapsule_Type")
    PyExc_BaseException = Ptr[cobj](dlsym(py_handle, "PyExc_BaseException", cobj))[0]
    PyExc_Exception = Ptr[cobj](dlsym(py_handle, "PyExc_Exception", cobj))[0]
    PyExc_NameError = Ptr[cobj](dlsym(py_handle, "PyExc_NameError", cobj))[0]
    PyExc_OSError = Ptr[cobj](dlsym(py_handle, "PyExc_OSError", cobj))[0]
    PyExc_IOError = Ptr[cobj](dlsym(py_handle, "PyExc_IOError", cobj))[0]
    PyExc_ValueError = Ptr[cobj](dlsym(py_handle, "PyExc_ValueError", cobj))[0]
    PyExc_LookupError = Ptr[cobj](dlsym(py_handle, "PyExc_LookupError", cobj))[0]
    PyExc_IndexError = Ptr[cobj](dlsym(py_handle, "PyExc_IndexError", cobj))[0]
    PyExc_KeyError = Ptr[cobj](dlsym(py_handle, "PyExc_KeyError", cobj))[0]
    PyExc_TypeError = Ptr[cobj](dlsym(py_handle, "PyExc_TypeError", cobj))[0]
    PyExc_ArithmeticError = Ptr[cobj](dlsym(py_handle, "PyExc_ArithmeticError", cobj))[0]
    PyExc_ZeroDivisionError = Ptr[cobj](dlsym(py_handle, "PyExc_ZeroDivisionError", cobj))[0]
    PyExc_OverflowError = Ptr[cobj](dlsym(py_handle, "PyExc_OverflowError", cobj))[0]
    PyExc_AttributeError = Ptr[cobj](dlsym(py_handle, "PyExc_AttributeError", cobj))[0]
    PyExc_RuntimeError = Ptr[cobj](dlsym(py_handle, "PyExc_RuntimeError", cobj))[0]
    PyExc_NotImplementedError = Ptr[cobj](dlsym(py_handle, "PyExc_NotImplementedError", cobj))[0]
    PyExc_StopIteration = Ptr[cobj](dlsym(py_handle, "PyExc_StopIteration", cobj))[0]
    PyExc_AssertionError = Ptr[cobj](dlsym(py_handle, "PyExc_AssertionError", cobj))[0]
    PyExc_SystemExit = Ptr[cobj](dlsym(py_handle, "PyExc_SystemExit", cobj))[0]

def init_handles_static():
    from C import Py_DecRef(cobj) as _Py_DecRef
    from C import Py_IncRef(cobj) as _Py_IncRef
    from C import Py_Initialize() as _Py_Initialize
    from C import PyImport_AddModule(cobj) -> cobj as _PyImport_AddModule
    from C import PyImport_AddModuleObject(cobj) -> cobj as _PyImport_AddModuleObject
    from C import PyImport_ImportModule(cobj) -> cobj as _PyImport_ImportModule
    from C import PyRun_SimpleString(cobj) as _PyRun_SimpleString
    from C import PyEval_GetGlobals() -> cobj as _PyEval_GetGlobals
    from C import PyEval_GetBuiltins() -> cobj as _PyEval_GetBuiltins
    from C import PyLong_AsLong(cobj) -> int as _PyLong_AsLong
    from C import PyLong_FromLong(int) -> cobj as _PyLong_FromLong
    from C import PyFloat_AsDouble(cobj) -> float as _PyFloat_AsDouble
    from C import PyFloat_FromDouble(float) -> cobj as _PyFloat_FromDouble
    from C import PyBool_FromLong(int) -> cobj as _PyBool_FromLong
    from C import PyBytes_AsString(cobj) -> cobj as _PyBytes_AsString
    from C import PyList_New(int) -> cobj as _PyList_New
    from C import PyList_Size(cobj) -> int as _PyList_Size
    from C import PyList_GetItem(cobj, int) -> cobj as _PyList_GetItem
    from C import PyList_SetItem(cobj, int, cobj) -> cobj as _PyList_SetItem
    from C import PyDict_New() -> cobj as _PyDict_New
    from C import PyDict_Next(cobj, Ptr[int], Ptr[cobj], Ptr[cobj]) -> int as _PyDict_Next
    from C import PyDict_GetItem(cobj, cobj) -> cobj as _PyDict_GetItem
    from C import PyDict_GetItemString(cobj, cobj) -> cobj as _PyDict_GetItemString
    from C import PyDict_SetItem(cobj, cobj, cobj) -> cobj as _PyDict_SetItem
    from C import PyDict_Size(cobj) -> int as _PyDict_Size
    from C import PySet_Add(cobj, cobj) -> cobj as _PySet_Add
    from C import PySet_New(cobj) -> cobj as _PySet_New
    from C import PyTuple_New(int) -> cobj as _PyTuple_New
    from C import PyTuple_Size(cobj) -> int as _PyTuple_Size
    from C import PyTuple_GetItem(cobj, int) -> cobj as _PyTuple_GetItem
    from C import PyTuple_SetItem(cobj, int, cobj) as _PyTuple_SetItem
    from C import PyUnicode_AsEncodedString(cobj, cobj, cobj) -> cobj as _PyUnicode_AsEncodedString
    from C import PyUnicode_DecodeFSDefaultAndSize(cobj, int) -> cobj as _PyUnicode_DecodeFSDefaultAndSize
    from C import PyUnicode_FromString(cobj) -> cobj as _PyUnicode_FromString
    from C import PyComplex_FromDoubles(float, float) -> cobj as _PyComplex_FromDoubles
    from C import PyComplex_RealAsDouble(cobj) -> float as _PyComplex_RealAsDouble
    from C import PyComplex_ImagAsDouble(cobj) -> float as _PyComplex_ImagAsDouble
    from C import PyIter_Next(cobj) -> cobj as _PyIter_Next
    from C import PySlice_New(cobj, cobj, cobj) -> cobj as _PySlice_New
    from C import PySlice_Unpack(cobj, Ptr[int], Ptr[int], Ptr[int]) -> int as _PySlice_Unpack
    from C import PyCapsule_New(cobj, cobj, cobj) -> cobj as _PyCapsule_New
    from C import PyCapsule_GetPointer(cobj, cobj) -> cobj as _PyCapsule_GetPointer
    from C import PyNumber_Add(cobj, cobj) -> cobj as _PyNumber_Add
    from C import PyNumber_Subtract(cobj, cobj) -> cobj as _PyNumber_Subtract
    from C import PyNumber_Multiply(cobj, cobj) -> cobj as _PyNumber_Multiply
    from C import PyNumber_MatrixMultiply(cobj, cobj) -> cobj as _PyNumber_MatrixMultiply
    from C import PyNumber_FloorDivide(cobj, cobj) -> cobj as _PyNumber_FloorDivide
    from C import PyNumber_TrueDivide(cobj, cobj) -> cobj as _PyNumber_TrueDivide
    from C import PyNumber_Remainder(cobj, cobj) -> cobj as _PyNumber_Remainder
    from C import PyNumber_Divmod(cobj, cobj) -> cobj as _PyNumber_Divmod
    from C import PyNumber_Power(cobj, cobj, cobj) -> cobj as _PyNumber_Power
    from C import PyNumber_Negative(cobj) -> cobj as _PyNumber_Negative
    from C import PyNumber_Positive(cobj) -> cobj as _PyNumber_Positive
    from C import PyNumber_Absolute(cobj) -> cobj as _PyNumber_Absolute
    from C import PyNumber_Invert(cobj) -> cobj as _PyNumber_Invert
    from C import PyNumber_Lshift(cobj, cobj) -> cobj as _PyNumber_Lshift
    from C import PyNumber_Rshift(cobj, cobj) -> cobj as _PyNumber_Rshift
    from C import PyNumber_And(cobj, cobj) -> cobj as _PyNumber_And
    from C import PyNumber_Xor(cobj, cobj) -> cobj as _PyNumber_Xor
    from C import PyNumber_Or(cobj, cobj) -> cobj as _PyNumber_Or
    from C import PyNumber_InPlaceAdd(cobj, cobj) -> cobj as _PyNumber_InPlaceAdd
    from C import PyNumber_InPlaceSubtract(cobj, cobj) -> cobj as _PyNumber_InPlaceSubtract
    from C import PyNumber_InPlaceMultiply(cobj, cobj) -> cobj as _PyNumber_InPlaceMultiply
    from C import PyNumber_InPlaceMatrixMultiply(cobj, cobj) -> cobj as _PyNumber_InPlaceMatrixMultiply
    from C import PyNumber_InPlaceFloorDivide(cobj, cobj) -> cobj as _PyNumber_InPlaceFloorDivide
    from C import PyNumber_InPlaceTrueDivide(cobj, cobj) -> cobj as _PyNumber_InPlaceTrueDivide
    from C import PyNumber_InPlaceRemainder(cobj, cobj) -> cobj as _PyNumber_InPlaceRemainder
    from C import PyNumber_InPlacePower(cobj, cobj, cobj) -> cobj as _PyNumber_InPlacePower
    from C import PyNumber_InPlaceLshift(cobj, cobj) -> cobj as _PyNumber_InPlaceLshift
    from C import PyNumber_InPlaceRshift(cobj, cobj) -> cobj as _PyNumber_InPlaceRshift
    from C import PyNumber_InPlaceAnd(cobj, cobj) -> cobj as _PyNumber_InPlaceAnd
    from C import PyNumber_InPlaceXor(cobj, cobj) -> cobj as _PyNumber_InPlaceXor
    from C import PyNumber_InPlaceOr(cobj, cobj) -> cobj as _PyNumber_InPlaceOr
    from C import PyNumber_Long(cobj) -> cobj as _PyNumber_Long
    from C import PyNumber_Float(cobj) -> cobj as _PyNumber_Float
    from C import PyNumber_Index(cobj) -> cobj as _PyNumber_Index
    from C import PyObject_Call(cobj, cobj, cobj) -> cobj as _PyObject_Call
    from C import PyObject_GetAttr(cobj, cobj) -> cobj as _PyObject_GetAttr
    from C import PyObject_GetAttrString(cobj, cobj) -> cobj as _PyObject_GetAttrString
    from C import PyObject_GetIter(cobj) -> cobj as _PyObject_GetIter
    from C import PyObject_HasAttrString(cobj, cobj) -> int as _PyObject_HasAttrString
    from C import PyObject_IsTrue(cobj) -> int as _PyObject_IsTrue
    from C import PyObject_Length(cobj) -> int as _PyObject_Length
    from C import PyObject_LengthHint(cobj, int) -> int as _PyObject_LengthHint
    from C import PyObject_SetAttrString(cobj, cobj, cobj) -> cobj as _PyObject_SetAttrString
    from C import PyObject_Str(cobj) -> cobj as _PyObject_Str
    from C import PyObject_Repr(cobj) -> cobj as _PyObject_Repr
    from C import PyObject_Hash(cobj) -> int as _PyObject_Hash
    from C import PyObject_GetItem(cobj, cobj) -> cobj as _PyObject_GetItem
    from C import PyObject_SetItem(cobj, cobj, cobj) -> int as _PyObject_SetItem
    from C import PyObject_DelItem(cobj, cobj) -> int as _PyObject_DelItem
    from C import PyObject_RichCompare(cobj, cobj, i32) -> cobj as _PyObject_RichCompare
    from C import PyObject_IsInstance(cobj, cobj) -> i32 as _PyObject_IsInstance
    from C import PyErr_Fetch(Ptr[cobj], Ptr[cobj], Ptr[cobj]) as _PyErr_Fetch
    from C import PyErr_NormalizeException(Ptr[cobj], Ptr[cobj], Ptr[cobj]) as _PyErr_NormalizeException
    from C import PyErr_SetString(cobj, cobj) as _PyErr_SetString
    from C import _Py_NoneStruct: cobj
    from C import _Py_TrueStruct: cobj
    from C import _Py_FalseStruct: cobj
    from C import _Py_EllipsisObject: cobj
    from C import _Py_NotImplementedStruct: cobj
    from C import PyLong_Type: cobj as _PyLong_Type
    from C import PyFloat_Type: cobj as _PyFloat_Type
    from C import PyBool_Type: cobj as _PyBool_Type
    from C import PyUnicode_Type: cobj as _PyUnicode_Type
    from C import PyComplex_Type: cobj as _PyComplex_Type
    from C import PyList_Type: cobj as _PyList_Type
    from C import PyDict_Type: cobj as _PyDict_Type
    from C import PySet_Type: cobj as _PySet_Type
    from C import PyTuple_Type: cobj as _PyTuple_Type
    from C import PySlice_Type: cobj as _PySlice_Type
    from C import PyCapsule_Type: cobj as _PyCapsule_Type
    from C import PyExc_BaseException: cobj as _PyExc_BaseException
    from C import PyExc_Exception: cobj as _PyExc_Exception
    from C import PyExc_NameError: cobj as _PyExc_NameError
    from C import PyExc_OSError: cobj as _PyExc_OSError
    from C import PyExc_IOError: cobj as _PyExc_IOError
    from C import PyExc_ValueError: cobj as _PyExc_ValueError
    from C import PyExc_LookupError: cobj as _PyExc_LookupError
    from C import PyExc_IndexError: cobj as _PyExc_IndexError
    from C import PyExc_KeyError: cobj as _PyExc_KeyError
    from C import PyExc_TypeError: cobj as _PyExc_TypeError
    from C import PyExc_ArithmeticError: cobj as _PyExc_ArithmeticError
    from C import PyExc_ZeroDivisionError: cobj as _PyExc_ZeroDivisionError
    from C import PyExc_OverflowError: cobj as _PyExc_OverflowError
    from C import PyExc_AttributeError: cobj as _PyExc_AttributeError
    from C import PyExc_RuntimeError: cobj as _PyExc_RuntimeError
    from C import PyExc_NotImplementedError: cobj as _PyExc_NotImplementedError
    from C import PyExc_StopIteration: cobj as _PyExc_StopIteration
    from C import PyExc_AssertionError: cobj as _PyExc_AssertionError
    from C import PyExc_SystemExit: cobj as _PyExc_SystemExit

    global Py_DecRef
    global Py_IncRef
    global Py_Initialize
    global PyImport_AddModule
    global PyImport_AddModuleObject
    global PyImport_ImportModule
    global PyRun_SimpleString
    global PyEval_GetGlobals
    global PyEval_GetBuiltins
    global PyLong_AsLong
    global PyLong_FromLong
    global PyFloat_AsDouble
    global PyFloat_FromDouble
    global PyBool_FromLong
    global PyBytes_AsString
    global PyList_New
    global PyList_Size
    global PyList_GetItem
    global PyList_SetItem
    global PyDict_New
    global PyDict_Next
    global PyDict_GetItem
    global PyDict_GetItemString
    global PyDict_SetItem
    global PyDict_Size
    global PySet_Add
    global PySet_New
    global PyTuple_New
    global PyTuple_Size
    global PyTuple_GetItem
    global PyTuple_SetItem
    global PyUnicode_AsEncodedString
    global PyUnicode_DecodeFSDefaultAndSize
    global PyUnicode_FromString
    global PyComplex_FromDoubles
    global PyComplex_RealAsDouble
    global PyComplex_ImagAsDouble
    global PyIter_Next
    global PySlice_New
    global PySlice_Unpack
    global PyCapsule_New
    global PyCapsule_GetPointer
    global PyNumber_Add
    global PyNumber_Subtract
    global PyNumber_Multiply
    global PyNumber_MatrixMultiply
    global PyNumber_FloorDivide
    global PyNumber_TrueDivide
    global PyNumber_Remainder
    global PyNumber_Divmod
    global PyNumber_Power
    global PyNumber_Negative
    global PyNumber_Positive
    global PyNumber_Absolute
    global PyNumber_Invert
    global PyNumber_Lshift
    global PyNumber_Rshift
    global PyNumber_And
    global PyNumber_Xor
    global PyNumber_Or
    global PyNumber_InPlaceAdd
    global PyNumber_InPlaceSubtract
    global PyNumber_InPlaceMultiply
    global PyNumber_InPlaceMatrixMultiply
    global PyNumber_InPlaceFloorDivide
    global PyNumber_InPlaceTrueDivide
    global PyNumber_InPlaceRemainder
    global PyNumber_InPlacePower
    global PyNumber_InPlaceLshift
    global PyNumber_InPlaceRshift
    global PyNumber_InPlaceAnd
    global PyNumber_InPlaceXor
    global PyNumber_InPlaceOr
    global PyNumber_Long
    global PyNumber_Float
    global PyNumber_Index
    global PyObject_Call
    global PyObject_GetAttr
    global PyObject_GetAttrString
    global PyObject_GetIter
    global PyObject_HasAttrString
    global PyObject_IsTrue
    global PyObject_Length
    global PyObject_LengthHint
    global PyObject_SetAttrString
    global PyObject_Str
    global PyObject_Repr
    global PyObject_Hash
    global PyObject_GetItem
    global PyObject_SetItem
    global PyObject_DelItem
    global PyObject_RichCompare
    global PyObject_IsInstance
    global PyErr_Fetch
    global PyErr_NormalizeException
    global PyErr_SetString
    global Py_None
    global Py_True
    global Py_False
    global Py_Ellipsis
    global Py_NotImplemented
    global PyLong_Type
    global PyFloat_Type
    global PyBool_Type
    global PyUnicode_Type
    global PyComplex_Type
    global PyList_Type
    global PyDict_Type
    global PySet_Type
    global PyTuple_Type
    global PySlice_Type
    global PyCapsule_Type
    global PyExc_BaseException
    global PyExc_Exception
    global PyExc_NameError
    global PyExc_OSError
    global PyExc_IOError
    global PyExc_ValueError
    global PyExc_LookupError
    global PyExc_IndexError
    global PyExc_KeyError
    global PyExc_TypeError
    global PyExc_ArithmeticError
    global PyExc_ZeroDivisionError
    global PyExc_OverflowError
    global PyExc_AttributeError
    global PyExc_RuntimeError
    global PyExc_NotImplementedError
    global PyExc_StopIteration
    global PyExc_AssertionError
    global PyExc_SystemExit

    Py_DecRef = _Py_DecRef
    Py_IncRef = _Py_IncRef
    Py_Initialize = _Py_Initialize
    PyImport_AddModule = _PyImport_AddModule
    PyImport_AddModuleObject = _PyImport_AddModuleObject
    PyImport_ImportModule = _PyImport_ImportModule
    PyRun_SimpleString = _PyRun_SimpleString
    PyEval_GetGlobals = _PyEval_GetGlobals
    PyEval_GetBuiltins = _PyEval_GetBuiltins
    PyLong_AsLong = _PyLong_AsLong
    PyLong_FromLong = _PyLong_FromLong
    PyFloat_AsDouble = _PyFloat_AsDouble
    PyFloat_FromDouble = _PyFloat_FromDouble
    PyBool_FromLong = _PyBool_FromLong
    PyBytes_AsString = _PyBytes_AsString
    PyList_New = _PyList_New
    PyList_Size = _PyList_Size
    PyList_GetItem = _PyList_GetItem
    PyList_SetItem = _PyList_SetItem
    PyDict_New = _PyDict_New
    PyDict_Next = _PyDict_Next
    PyDict_GetItem = _PyDict_GetItem
    PyDict_GetItemString = _PyDict_GetItemString
    PyDict_SetItem = _PyDict_SetItem
    PyDict_Size = _PyDict_Size
    PySet_Add = _PySet_Add
    PySet_New = _PySet_New
    PyTuple_New = _PyTuple_New
    PyTuple_Size = _PyTuple_Size
    PyTuple_GetItem = _PyTuple_GetItem
    PyTuple_SetItem = _PyTuple_SetItem
    PyUnicode_AsEncodedString = _PyUnicode_AsEncodedString
    PyUnicode_DecodeFSDefaultAndSize = _PyUnicode_DecodeFSDefaultAndSize
    PyUnicode_FromString = _PyUnicode_FromString
    PyComplex_FromDoubles = _PyComplex_FromDoubles
    PyComplex_RealAsDouble = _PyComplex_RealAsDouble
    PyComplex_ImagAsDouble = _PyComplex_ImagAsDouble
    PyIter_Next = _PyIter_Next
    PySlice_New = _PySlice_New
    PySlice_Unpack = _PySlice_Unpack
    PyCapsule_New = _PyCapsule_New
    PyCapsule_GetPointer = _PyCapsule_GetPointer
    PyNumber_Add = _PyNumber_Add
    PyNumber_Subtract = _PyNumber_Subtract
    PyNumber_Multiply = _PyNumber_Multiply
    PyNumber_MatrixMultiply = _PyNumber_MatrixMultiply
    PyNumber_FloorDivide = _PyNumber_FloorDivide
    PyNumber_TrueDivide = _PyNumber_TrueDivide
    PyNumber_Remainder = _PyNumber_Remainder
    PyNumber_Divmod = _PyNumber_Divmod
    PyNumber_Power = _PyNumber_Power
    PyNumber_Negative = _PyNumber_Negative
    PyNumber_Positive = _PyNumber_Positive
    PyNumber_Absolute = _PyNumber_Absolute
    PyNumber_Invert = _PyNumber_Invert
    PyNumber_Lshift = _PyNumber_Lshift
    PyNumber_Rshift = _PyNumber_Rshift
    PyNumber_And = _PyNumber_And
    PyNumber_Xor = _PyNumber_Xor
    PyNumber_Or = _PyNumber_Or
    PyNumber_InPlaceAdd = _PyNumber_InPlaceAdd
    PyNumber_InPlaceSubtract = _PyNumber_InPlaceSubtract
    PyNumber_InPlaceMultiply = _PyNumber_InPlaceMultiply
    PyNumber_InPlaceMatrixMultiply = _PyNumber_InPlaceMatrixMultiply
    PyNumber_InPlaceFloorDivide = _PyNumber_InPlaceFloorDivide
    PyNumber_InPlaceTrueDivide = _PyNumber_InPlaceTrueDivide
    PyNumber_InPlaceRemainder = _PyNumber_InPlaceRemainder
    PyNumber_InPlacePower = _PyNumber_InPlacePower
    PyNumber_InPlaceLshift = _PyNumber_InPlaceLshift
    PyNumber_InPlaceRshift = _PyNumber_InPlaceRshift
    PyNumber_InPlaceAnd = _PyNumber_InPlaceAnd
    PyNumber_InPlaceXor = _PyNumber_InPlaceXor
    PyNumber_InPlaceOr = _PyNumber_InPlaceOr
    PyNumber_Long = _PyNumber_Long
    PyNumber_Float = _PyNumber_Float
    PyNumber_Index = _PyNumber_Index
    PyObject_Call = _PyObject_Call
    PyObject_GetAttr = _PyObject_GetAttr
    PyObject_GetAttrString = _PyObject_GetAttrString
    PyObject_GetIter = _PyObject_GetIter
    PyObject_HasAttrString = _PyObject_HasAttrString
    PyObject_IsTrue = _PyObject_IsTrue
    PyObject_Length = _PyObject_Length
    PyObject_LengthHint = _PyObject_LengthHint
    PyObject_SetAttrString = _PyObject_SetAttrString
    PyObject_Str = _PyObject_Str
    PyObject_Repr = _PyObject_Repr
    PyObject_Hash = _PyObject_Hash
    PyObject_GetItem = _PyObject_GetItem
    PyObject_SetItem = _PyObject_SetItem
    PyObject_DelItem = _PyObject_DelItem
    PyObject_RichCompare = _PyObject_RichCompare
    PyObject_IsInstance = _PyObject_IsInstance
    PyErr_Fetch = _PyErr_Fetch
    PyErr_NormalizeException = _PyErr_NormalizeException
    PyErr_SetString = _PyErr_SetString
    Py_None = __ptr__(_Py_NoneStruct).as_byte()
    Py_True = __ptr__(_Py_TrueStruct).as_byte()
    Py_False = __ptr__(_Py_FalseStruct).as_byte()
    Py_Ellipsis = __ptr__(_Py_EllipsisObject).as_byte()
    Py_NotImplemented = __ptr__(_Py_NotImplementedStruct).as_byte()
    PyLong_Type = __ptr__(_PyLong_Type).as_byte()
    PyFloat_Type = __ptr__(_PyFloat_Type).as_byte()
    PyBool_Type = __ptr__(_PyBool_Type).as_byte()
    PyUnicode_Type = __ptr__(_PyUnicode_Type).as_byte()
    PyComplex_Type = __ptr__(_PyComplex_Type).as_byte()
    PyList_Type = __ptr__(_PyList_Type).as_byte()
    PyDict_Type = __ptr__(_PyDict_Type).as_byte()
    PySet_Type = __ptr__(_PySet_Type).as_byte()
    PyTuple_Type = __ptr__(_PyTuple_Type).as_byte()
    PySlice_Type = __ptr__(_PySlice_Type).as_byte()
    PyCapsule_Type = __ptr__(_PyCapsule_Type).as_byte()
    PyExc_BaseException = _PyExc_BaseException
    PyExc_Exception = _PyExc_Exception
    PyExc_NameError = _PyExc_NameError
    PyExc_OSError = _PyExc_OSError
    PyExc_IOError = _PyExc_IOError
    PyExc_ValueError = _PyExc_ValueError
    PyExc_LookupError = _PyExc_LookupError
    PyExc_IndexError = _PyExc_IndexError
    PyExc_KeyError = _PyExc_KeyError
    PyExc_TypeError = _PyExc_TypeError
    PyExc_ArithmeticError = _PyExc_ArithmeticError
    PyExc_ZeroDivisionError = _PyExc_ZeroDivisionError
    PyExc_OverflowError = _PyExc_OverflowError
    PyExc_AttributeError = _PyExc_AttributeError
    PyExc_RuntimeError = _PyExc_RuntimeError
    PyExc_NotImplementedError = _PyExc_NotImplementedError
    PyExc_StopIteration = _PyExc_StopIteration
    PyExc_AssertionError = _PyExc_AssertionError
    PyExc_SystemExit = _PyExc_SystemExit

def init_error_py_types():
    BaseException._pytype = PyExc_BaseException
    Exception._pytype = PyExc_Exception
    NameError._pytype = PyExc_NameError
    OSError._pytype = PyExc_OSError
    IOError._pytype = PyExc_IOError
    ValueError._pytype = PyExc_ValueError
    LookupError._pytype = PyExc_LookupError
    IndexError._pytype = PyExc_IndexError
    KeyError._pytype = PyExc_KeyError
    TypeError._pytype = PyExc_TypeError
    ArithmeticError._pytype = PyExc_ArithmeticError
    ZeroDivisionError._pytype = PyExc_ZeroDivisionError
    OverflowError._pytype = PyExc_OverflowError
    AttributeError._pytype = PyExc_AttributeError
    RuntimeError._pytype = PyExc_RuntimeError
    NotImplementedError._pytype = PyExc_NotImplementedError
    StopIteration._pytype = PyExc_StopIteration
    AssertionError._pytype = PyExc_AssertionError
    SystemExit._pytype = PyExc_SystemExit

def setup_python(python_loaded: bool):
    global _PY_INITIALIZED
    if _PY_INITIALIZED:
        return

    py_handle = cobj()
    if python_loaded:
        py_handle = dlopen("", RTLD_LOCAL | RTLD_NOW)
    else:
        LD = os.getenv("HERCULES_PYTHON", default="libpython." + dlext())
        py_handle = dlopen(LD, RTLD_LOCAL | RTLD_NOW)

    init_handles_dlopen(py_handle)
    init_error_py_types()

    if not python_loaded:
        Py_Initialize()

    _PY_INITIALIZED = True

def ensure_initialized(python_loaded: bool = False):
    if __py_extension__:
        init_handles_static()
        init_error_py_types()
    else:
        setup_python(python_loaded)
        PyRun_SimpleString(_PY_INIT.c_str())

def setup_decorator():
    setup_python(True)

@tuple
class _PyArg_Parser:
    initialized: i32
    format: cobj
    keywords: Ptr[cobj]
    fname: cobj
    custom_msg: cobj
    pos: i32
    min: i32
    max: i32
    kwtuple: cobj
    next: cobj

    def __new__(fname: cobj, keywords: Ptr[cobj], format: cobj):
        z = i32(0)
        o = cobj()
        return _PyArg_Parser(z, format, keywords, fname, o, z, z, z, o, o)

@extend
class pyobj:
    def __new__() -> pyobj:
        return __internal__.class_alloc(pyobj)

    def __raw__(self) -> Ptr[byte]:
        return __internal__.class_raw(self)

    def __init__(self, p: Ptr[byte], steal: bool = False):
        self.p = p
        if not steal:
            self.incref()

    def __del__(self):
        self.decref()

    def _getattr(self, name: str) -> pyobj:
        return pyobj(pyobj.exc_wrap(PyObject_GetAttrString(self.p, name.c_str())), steal=True)

    def __add__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Add(self.p, other.__to_py__())), steal=True)

    def __radd__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Add(other.__to_py__(), self.p)), steal=True)

    def __sub__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Subtract(self.p, other.__to_py__())), steal=True)

    def __rsub__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Subtract(other.__to_py__(), self.p)), steal=True)

    def __mul__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Multiply(self.p, other.__to_py__())), steal=True)

    def __rmul__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Multiply(other.__to_py__(), self.p)), steal=True)

    def __matmul__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_MatrixMultiply(self.p, other.__to_py__())), steal=True)

    def __rmatmul__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_MatrixMultiply(other.__to_py__(), self.p)), steal=True)

    def __floordiv__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_FloorDivide(self.p, other.__to_py__())), steal=True)

    def __rfloordiv__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_FloorDivide(other.__to_py__(), self.p)), steal=True)

    def __truediv__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_TrueDivide(self.p, other.__to_py__())), steal=True)

    def __rtruediv__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_TrueDivide(other.__to_py__(), self.p)), steal=True)

    def __mod__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Remainder(self.p, other.__to_py__())), steal=True)

    def __rmod__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Remainder(other.__to_py__(), self.p)), steal=True)

    def __divmod__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Divmod(self.p, other.__to_py__())), steal=True)

    def __rdivmod__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Divmod(other.__to_py__(), self.p)), steal=True)

    def __pow__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Power(self.p, other.__to_py__(), Py_None)), steal=True)

    def __rpow__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Power(other.__to_py__(), self.p, Py_None)), steal=True)

    def __neg__(self):
        return pyobj(pyobj.exc_wrap(PyNumber_Negative(self.p)), steal=True)

    def __pos__(self):
        return pyobj(pyobj.exc_wrap(PyNumber_Positive(self.p)), steal=True)

    def __invert__(self):
        return pyobj(pyobj.exc_wrap(PyNumber_Invert(self.p)), steal=True)

    def __lshift__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Lshift(self.p, other.__to_py__())), steal=True)

    def __rlshift__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Lshift(other.__to_py__(), self.p)), steal=True)

    def __rshift__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Rshift(self.p, other.__to_py__())), steal=True)

    def __rrshift__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Rshift(other.__to_py__(), self.p)), steal=True)

    def __and__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_And(self.p, other.__to_py__())), steal=True)

    def __rand__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_And(other.__to_py__(), self.p)), steal=True)

    def __xor__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Xor(self.p, other.__to_py__())), steal=True)

    def __rxor__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Xor(other.__to_py__(), self.p)), steal=True)

    def __or__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Or(self.p, other.__to_py__())), steal=True)

    def __ror__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_Or(other.__to_py__(), self.p)), steal=True)

    def __iadd__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceAdd(self.p, other.__to_py__())), steal=True)

    def __isub__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceSubtract(self.p, other.__to_py__())), steal=True)

    def __imul__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceMultiply(self.p, other.__to_py__())), steal=True)

    def __imatmul__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceMatrixMultiply(self.p, other.__to_py__())), steal=True)

    def __ifloordiv__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceFloorDivide(self.p, other.__to_py__())), steal=True)

    def __itruediv__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceTrueDivide(self.p, other.__to_py__())), steal=True)

    def __imod__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceRemainder(self.p, other.__to_py__())), steal=True)

    def __ipow__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlacePower(self.p, other.__to_py__(), Py_None)), steal=True)

    def __ilshift__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceLshift(self.p, other.__to_py__())), steal=True)

    def __irshift__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceRshift(self.p, other.__to_py__())), steal=True)

    def __iand__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceAnd(self.p, other.__to_py__())), steal=True)

    def __ixor__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceXor(self.p, other.__to_py__())), steal=True)

    def __ior__(self, other):
        return pyobj(pyobj.exc_wrap(PyNumber_InPlaceOr(self.p, other.__to_py__())), steal=True)

    def __int__(self):
        o = pyobj.exc_wrap(PyNumber_Long(self.p))
        x = int.__from_py__(o)
        pyobj.decref(o)
        return x

    def __float__(self):
        o = pyobj.exc_wrap(PyNumber_Float(self.p))
        x = float.__from_py__(o)
        pyobj.decref(o)
        return x

    def __index__(self):
        o = pyobj.exc_wrap(PyNumber_Index(self.p))
        x = int.__from_py__(o)
        pyobj.decref(o)
        return x

    def __len__(self) -> int:
        return pyobj.exc_wrap(PyObject_Length(self.p))

    def __length_hint__(self) -> int:
        return pyobj.exc_wrap(PyObject_LengthHint(self.p))

    def __getitem__(self, key):
        return pyobj(pyobj.exc_wrap(PyObject_GetItem(self.p, key.__to_py__())), steal=True)

    def __setitem__(self, key, v):
        pyobj.exc_wrap(PyObject_SetItem(self.p, key.__to_py__(), v.__to_py__()))

    def __delitem__(self, key):
        return pyobj.exc_wrap(PyObject_DelItem(self.p, key.__to_py__()))

    def __lt__(self, other):
        return pyobj(pyobj.exc_wrap(PyObject_RichCompare(self.p, other.__to_py__(), i32(Py_LT))), steal=True)

    def __le__(self, other):
        return pyobj(pyobj.exc_wrap(PyObject_RichCompare(self.p, other.__to_py__(), i32(Py_LE))), steal=True)

    def __eq__(self, other):
        return pyobj(pyobj.exc_wrap(PyObject_RichCompare(self.p, other.__to_py__(), i32(Py_EQ))), steal=True)

    def __ne__(self, other):
        return pyobj(pyobj.exc_wrap(PyObject_RichCompare(self.p, other.__to_py__(), i32(Py_NE))), steal=True)

    def __gt__(self, other):
        return pyobj(pyobj.exc_wrap(PyObject_RichCompare(self.p, other.__to_py__(), i32(Py_GT))), steal=True)

    def __ge__(self, other):
        return pyobj(pyobj.exc_wrap(PyObject_RichCompare(self.p, other.__to_py__(), i32(Py_GE))), steal=True)

    def __to_py__(self) -> cobj:
        return self.p

    def __from_py__(p: cobj) -> pyobj:
        return pyobj(p)

    def __str__(self) -> str:
        o = pyobj.exc_wrap(PyObject_Str(self.p))
        return pyobj.exc_wrap(str.__from_py__(o))

    def __repr__(self) -> str:
        o = pyobj.exc_wrap(PyObject_Repr(self.p))
        return pyobj.exc_wrap(str.__from_py__(o))

    def __hash__(self) -> int:
        return pyobj.exc_wrap(PyObject_Hash(self.p))

    def __iter__(self) -> Generator[pyobj]:
        it = PyObject_GetIter(self.p)
        if not it:
            raise TypeError("Python object is not iterable")
        try:
            while i := PyIter_Next(it):
                yield pyobj(pyobj.exc_wrap(i), steal=True)
        finally:
            pyobj.decref(it)
        pyobj.exc_check()

    def to_str(self, errors: str, empty: str = "") -> str:
        return pyobj.to_str(self.p, errors, empty)

    def to_str(p: cobj, errors: str, empty: str = "") -> str:
        obj = PyUnicode_AsEncodedString(p, "utf-8".c_str(), errors.c_str())
        if obj == cobj():
            return empty
        bts = PyBytes_AsString(obj)
        res = str.from_ptr(bts)
        pyobj.decref(obj)
        return res

    def exc_check():
        ptype, pvalue, ptraceback = cobj(), cobj(), cobj()
        PyErr_Fetch(__ptr__(ptype), __ptr__(pvalue), __ptr__(ptraceback))
        PyErr_NormalizeException(__ptr__(ptype), __ptr__(pvalue), __ptr__(ptraceback))
        if ptype != cobj():
            py_msg = PyObject_Str(pvalue) if pvalue != cobj() else pvalue
            msg = pyobj.to_str(py_msg, "ignore", "<empty Python message>")

            pyobj.decref(ptype)
            pyobj.decref(ptraceback)
            pyobj.decref(py_msg)

            # pyobj.decref(pvalue)
            raise PyError(msg, pyobj(pvalue))

    def exc_wrap(_retval: T, T: type) -> T:
        pyobj.exc_check()
        return _retval

    def incref(self):
        Py_IncRef(self.p)
        return self

    def incref(ptr: Ptr[byte]):
        Py_IncRef(ptr)

    def decref(self):
        Py_DecRef(self.p)
        return self

    def decref(ptr: Ptr[byte]):
        Py_DecRef(ptr)

    def __call__(self, *args, **kwargs):
        args_py = args.__to_py__()
        kws_py = cobj()
        if staticlen(kwargs) > 0:
            names = iter(kwargs.__dict__())
            kws = {next(names): pyobj(i.__to_py__(), steal=True) for i in kwargs}
            kws_py = kws.__to_py__()
        return pyobj(pyobj.exc_wrap(PyObject_Call(self.p, args_py, kws_py)), steal=True)

    def _tuple_new(length: int):
        return pyobj.exc_wrap(PyTuple_New(length))

    def _tuple_size(p: cobj):
        return pyobj.exc_wrap(PyTuple_Size(p))

    def _tuple_set(p: cobj, idx: int, val: cobj):
        PyTuple_SetItem(p, idx, val)
        pyobj.exc_check()

    def _tuple_get(p: cobj, idx: int) -> cobj:
        return pyobj.exc_wrap(PyTuple_GetItem(p, idx))

    def _import(name: str) -> pyobj:
        ensure_initialized()
        if name in _PY_MODULE_CACHE:
            return _PY_MODULE_CACHE[name]
        m = pyobj(pyobj.exc_wrap(PyImport_ImportModule(name.c_str())), steal=True)
        _PY_MODULE_CACHE[name] = m
        return m

    def _exec(code: str):
        ensure_initialized()
        PyRun_SimpleString(code.c_str())

    def _globals() -> pyobj:
        p = PyEval_GetGlobals()
        if p == cobj():
            Py_IncRef(Py_None)
            return pyobj(Py_None)
        return pyobj(p)

    def _builtins() -> pyobj:
        return pyobj(PyEval_GetBuiltins())

    def _get_module(name: str) -> pyobj:
        p = pyobj(pyobj.exc_wrap(PyImport_AddModule(name.c_str())))
        return p

    def _main_module() -> pyobj:
        return pyobj._get_module("__main__")

    def _repr_mimebundle_(self, bundle=Set[str]()) -> Dict[str, str]:
        fn = pyobj._main_module()._getattr("__hercules_repr__")
        assert fn.p != cobj(), "cannot find python.__hercules_repr__"
        mime, txt = Tuple[str, str].__from_py__(fn.__call__(self).p)
        return {mime: txt}

    def __bool__(self):
        return bool(pyobj.exc_wrap(PyObject_IsTrue(self.p) == 1))

def _get_identifier(typ: str) -> pyobj:
    t = pyobj._builtins()[typ]
    if t.p == cobj():
        t = pyobj._main_module()[typ]
    return t

def _isinstance(what: pyobj, typ: pyobj) -> bool:
    return bool(pyobj.exc_wrap(PyObject_IsInstance(what.p, typ.p)))

@tuple
class _PyObject_Struct:
    refcnt: int
    pytype: cobj

def _conversion_error(name: Static[str]):
    raise PyError("conversion error: Python object did not have type '" + name + "'")

def _ensure_type(o: cobj, t: cobj, name: Static[str]):
    if Ptr[_PyObject_Struct](o)[0].pytype != t:
        _conversion_error(name)


# Type conversions

@extend
class NoneType:
    def __to_py__(self) -> cobj:
        Py_IncRef(Py_None)
        return Py_None

    def __from_py__(x: cobj) -> None:
        if x != Py_None:
            _conversion_error("NoneType")
        return

@extend
class int:
    def __to_py__(self) -> cobj:
        return pyobj.exc_wrap(PyLong_FromLong(self))

    def __from_py__(i: cobj) -> int:
        _ensure_type(i, PyLong_Type, "int")
        return PyLong_AsLong(i)

@extend
class float:
    def __to_py__(self) -> cobj:
        return pyobj.exc_wrap(PyFloat_FromDouble(self))

    def __from_py__(d: cobj) -> float:
        return pyobj.exc_wrap(PyFloat_AsDouble(d))

@extend
class bool:
    def __to_py__(self) -> cobj:
        return pyobj.exc_wrap(PyBool_FromLong(int(self)))

    def __from_py__(b: cobj) -> bool:
        _ensure_type(b, PyBool_Type, "bool")
        return PyObject_IsTrue(b) != 0

@extend
class byte:
    def __to_py__(self) -> cobj:
        return str.__to_py__(str(__ptr__(self), 1))

    def __from_py__(c: cobj) -> byte:
        return str.__from_py__(c).ptr[0]

@extend
class str:
    def __to_py__(self) -> cobj:
        return pyobj.exc_wrap(PyUnicode_DecodeFSDefaultAndSize(self.ptr, self.len))

    def __from_py__(s: cobj) -> str:
        return pyobj.exc_wrap(pyobj.to_str(s, "strict"))

@extend
class complex:
    def __to_py__(self) -> cobj:
        return pyobj.exc_wrap(PyComplex_FromDoubles(self.real, self.imag))

    def __from_py__(c: cobj) -> complex:
        _ensure_type(c, PyComplex_Type, "complex")
        real = PyComplex_RealAsDouble(c)
        imag = PyComplex_ImagAsDouble(c)
        return complex(real, imag)

@extend
class List:
    def __to_py__(self) -> cobj:
        pylist = PyList_New(len(self))
        pyobj.exc_check()
        idx = 0
        for a in self:
            PyList_SetItem(pylist, idx, a.__to_py__())
            pyobj.exc_check()
            idx += 1
        return pylist

    def __from_py__(v: cobj) -> List[T]:
        _ensure_type(v, PyList_Type, "list")
        n = PyList_Size(v)
        t = List[T](n)
        for i in range(n):
            elem = PyList_GetItem(v, i)
            t.append(T.__from_py__(elem))
        return t

@extend
class Dict:
    def __to_py__(self) -> cobj:
        pydict = PyDict_New()
        pyobj.exc_check()
        for k, v in self.items():
            PyDict_SetItem(pydict, k.__to_py__(), v.__to_py__())
            pyobj.exc_check()
        return pydict

    def __from_py__(d: cobj) -> Dict[K, V]:
        _ensure_type(d, PyDict_Type, "dict")
        b = dict[K, V]()
        pos = 0
        k_ptr = cobj()
        v_ptr = cobj()
        while PyDict_Next(d, __ptr__(pos), __ptr__(k_ptr), __ptr__(v_ptr)):
            k = K.__from_py__(k_ptr)
            v = V.__from_py__(v_ptr)
            b[k] = v
        return b

@extend
class Set:
    def __to_py__(self) -> cobj:
        pyset = PySet_New(cobj())
        pyobj.exc_check()
        for a in self:
            PySet_Add(pyset, a.__to_py__())
            pyobj.exc_check()
        return pyset

    def __from_py__(s: cobj) -> Set[K]:
        _ensure_type(s, PySet_Type, "set")
        b = set[K]()
        s_iter = PyObject_GetIter(s)
        while True:
            k_ptr = pyobj.exc_wrap(PyIter_Next(s_iter))
            if not k_ptr:
                break
            k = K.__from_py__(k_ptr)
            pyobj.decref(k_ptr)
            b.add(k)
        pyobj.decref(s_iter)
        return b

@extend
class DynamicTuple:
    def __to_py__(self) -> cobj:
        pytup = PyTuple_New(len(self))
        i = 0
        for a in self:
            PyTuple_SetItem(pytup, i, a.__to_py__())
            pyobj.exc_check()
            i += 1
        return pytup

    def __from_py__(t: cobj) -> DynamicTuple[T]:
        _ensure_type(t, PyTuple_Type, "tuple")
        n = PyTuple_Size(t)
        p = Ptr[T](n)
        for i in range(n):
            p[i] = T.__from_py__(PyTuple_GetItem(t, i))
        return DynamicTuple(p, n)

@extend
class Slice:
    def __to_py__(self) -> cobj:
        start = self.start
        stop = self.stop
        step = self.step
        start_py = start.__to_py__() if start is not None else cobj()
        stop_py = stop.__to_py__() if stop is not None else cobj()
        step_py = step.__to_py__() if step is not None else cobj()
        return PySlice_New(start_py, stop_py, step_py)

    def __from_py__(s: cobj) -> Slice:
        _ensure_type(s, PySlice_Type, "slice")
        start = 0
        stop = 0
        step = 0
        PySlice_Unpack(s, __ptr__(start), __ptr__(stop), __ptr__(step))
        return Slice(Optional(start), Optional(stop), Optional(step))

@extend
class Optional:
    def __to_py__(self) -> cobj:
        if self is None:
            return Py_None
        else:
            return self.__val__().__to_py__()

    def __from_py__(o: cobj) -> Optional[T]:
        if o == Py_None:
            return Optional[T]()
        else:
            return Optional[T](T.__from_py__(o))

@extend
class ellipsis:
    def __to_py__(self) -> cobj:
        return Py_Ellipsis

    def __from_py__(e: cobj) -> ellipsis:
        if e == Py_Ellipsis:
            return Ellipsis
        else:
            _conversion_error("ellipsis")

__pyenv__: Optional[pyobj] = None
def _____(): __pyenv__  # make it global!


import internal.static as _S


class _PyWrapError(Static[PyError]):
    def __init__(self, message: str, pytype: pyobj = pyobj(cobj(), steal=True)):
        super().__init__("_PyWrapError", message)
        self.pytype = pytype

    def __init__(self, e: PyError):
        self.__init__("_PyWrapError", e.message, e.pytype)


class _PyWrap:
    def _dispatch_error(F: Static[str]):
        raise TypeError("could not find callable method '" + F + "' for given arguments")

    def _wrap(args, T: type, F: Static[str], map):
        for fn in _S.fn_overloads(T, F):
            a = _PyWrap._args_from_py(fn, args)
            if a is None:
                continue
            if _S.fn_can_call(fn, *a):
                try:
                    return map(fn, a)
                except PyError as e:
                    pass
        _PyWrap._dispatch_error(F)

    def _wrap_unary(obj: cobj, T: type, F: Static[str]) -> cobj:
        return _PyWrap._wrap(
            (obj,), T=T, F=F,
            map=lambda f, a: f(*a).__to_py__()
        )

    def _args_from_py(fn, args):
        def err(fail: Ptr[bool], T: type = NoneType) -> T:
            fail[0] = True
            # auto-return zero-initialized T

        def get_arg(F, p, k, fail: Ptr[bool], i: Static[int]):
            if _S.fn_arg_has_type(F, i):
                return _S.fn_arg_get_type(F, i).__from_py__(p[i]) if p[i] != cobj() else (
                    _S.fn_get_default(F, i) if _S.fn_has_default(F, i)
                    else err(fail, _S.fn_arg_get_type(F, i))
                )
            else:
                return pyobj(p[i], steal=False) if p[i] != cobj() else (
                    _S.fn_get_default(F, i) if _S.fn_has_default(F, i) else err(fail)
                )

        fail = False
        pargs = Ptr[cobj](__ptr__(args).as_byte())
        try:
            ta = tuple(get_arg(fn, pargs, k, __ptr__(fail), i) for i, k in staticenumerate(_S.fn_args(fn)))
            if fail:
                return None
            return _S.fn_wrap_call_args(fn, *ta)
        except PyError:
            return None

    def _reorder_args(fn, self: cobj, args: cobj, kwargs: cobj, M: Static[int] = 1):
        nargs = PyTuple_Size(args)
        nkwargs = PyDict_Size(kwargs) if kwargs != cobj() else 0

        args_ordered = tuple(cobj() for _ in _S.fn_args(fn))
        pargs = Ptr[cobj](__ptr__(args_ordered).as_byte())

        if nargs + nkwargs + M > len(args_ordered):
            return None

        if M:
            pargs[0] = self

        for i in range(nargs):
            pargs[i + M] = PyTuple_GetItem(args, i)

        kwused = 0
        for i, k in staticenumerate(_S.fn_args(fn)):
            if i < nargs + M:
                continue

            p = PyDict_GetItemString(kwargs, k.ptr) if nkwargs else cobj()
            if p != cobj():
                pargs[i] = p
                kwused += 1

        if kwused != nkwargs:
            return None

        return _PyWrap._args_from_py(fn, args_ordered)

    def _reorder_args_fastcall(
        fn, self: cobj, args: Ptr[cobj], nargs: int,
        kwds: Ptr[str], nkw: int, M: Static[int] = 1
    ):
        args_ordered = tuple(cobj() for _ in _S.fn_args(fn))
        pargs = Ptr[cobj](__ptr__(args_ordered).as_byte())

        if nargs + M > len(args_ordered):
            return None

        if M:
            pargs[0] = self

        for i in range(nargs):
            pargs[i + M] = args[i]

        for i in range(nargs, nargs + nkw):
            kw = kwds[i - nargs]
            o = args[i]

            found = False
            j = M
            for i, k in staticenumerate(_S.fn_args(fn)):
                if M and i == 0:
                    continue
                if kw == k:
                    if not pargs[j]:
                        pargs[j] = o
                    else:
                        return None
                    found = True
                    break
                j += 1
            if not found:
                return None

        return _PyWrap._args_from_py(fn, args_ordered)

    def wrap_magic_abs(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__abs__")

    def wrap_magic_pos(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__pos__")

    def wrap_magic_neg(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__neg__")

    def wrap_magic_invert(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__invert__")

    def wrap_magic_int(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__int__")

    def wrap_magic_float(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__float__")

    def wrap_magic_index(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__index__")

    def wrap_magic_repr(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__repr__")

    def wrap_magic_str(obj: cobj, T: type):
        return _PyWrap._wrap_unary(obj, T, "__str__")

    def _wrap_binary(obj: cobj, obj2: cobj, T: type, F: Static[str]) -> cobj:
        return _PyWrap._wrap(
            (obj, obj2), T=T, F=F,
            map=lambda f, a: f(*a).__to_py__()
        )

    def wrap_magic_add(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__add__")

    def wrap_magic_radd(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__radd__")

    def wrap_magic_iadd(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__iadd__")

    def wrap_magic_sub(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__sub__")

    def wrap_magic_rsub(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rsub__")

    def wrap_magic_isub(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__isub__")

    def wrap_magic_mul(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__mul__")

    def wrap_magic_rmul(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rmul__")

    def wrap_magic_imul(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__imul__")

    def wrap_magic_mod(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__mod__")

    def wrap_magic_rmod(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rmod__")

    def wrap_magic_imod(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__imod__")

    def wrap_magic_divmod(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__divmod__")

    def wrap_magic_rdivmod(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rdivmod__")

    def wrap_magic_lshift(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__lshift__")

    def wrap_magic_rlshift(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rlshift__")

    def wrap_magic_ilshift(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__ilshift__")

    def wrap_magic_rshift(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rshift__")

    def wrap_magic_rrshift(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rrshift__")

    def wrap_magic_irshift(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__irshift__")

    def wrap_magic_and(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__and__")

    def wrap_magic_rand(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rand__")

    def wrap_magic_iand(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__iand__")

    def wrap_magic_xor(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__xor__")

    def wrap_magic_rxor(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rxor__")

    def wrap_magic_ixor(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__ixor__")

    def wrap_magic_or(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__or__")

    def wrap_magic_ror(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__ror__")

    def wrap_magic_ior(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__ior__")

    def wrap_magic_floordiv(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__floordiv__")

    def wrap_magic_ifloordiv(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__ifloordiv__")

    def wrap_magic_truediv(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__truediv__")

    def wrap_magic_itruediv(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__itruediv__")

    def wrap_magic_matmul(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__matmul__")

    def wrap_magic_rmatmul(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rmatmul__")

    def wrap_magic_imatmul(obj: cobj, obj2: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__imatmul__")

    def wrap_magic_pow(obj: cobj, obj2: cobj, obj3: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__pow__")

    def wrap_magic_rpow(obj: cobj, obj2: cobj, obj3: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__rpow__")

    def wrap_magic_ipow(obj: cobj, obj2: cobj, obj3: cobj, T: type):
        return _PyWrap._wrap_binary(obj, obj2, T, "__ipow__")

    def _wrap_hash(obj: cobj, T: type, F: Static[str]) -> i64:
        return _PyWrap._wrap(
            (obj,), T=T, F=F,
            map=lambda f, a: f(*a)
        )

    def wrap_magic_len(obj: cobj, T: type):
        return _PyWrap._wrap_hash(obj, T, "__len__")

    def wrap_magic_hash(obj: cobj, T: type):
        return _PyWrap._wrap_hash(obj, T, "__hash__")

    def wrap_magic_bool(obj: cobj, T: type) -> i32:
        return _PyWrap._wrap(
            (obj,), T=T, F="__bool__",
            map=lambda f, a: i32(1) if f(*a) else i32(0)
        )

    def wrap_magic_del(obj: cobj, T: type):
        _PyWrap._wrap(
            (obj,), T=T, F="__del__",
            map=lambda f, a: f(*a)
        )

    def wrap_magic_contains(obj: cobj, arg: cobj, T: type) -> i32:
        return _PyWrap._wrap(
            (obj, arg,), T=T, F="__contains__",
            map=lambda f, a: i32(1) if f(*a) else i32(0)
        )

    def wrap_magic_init(obj: cobj, args: cobj, kwargs: cobj, T: type) -> i32:
        if isinstance(T, ByRef):
            F: Static[str] = "__init__"
            for fn in _S.fn_overloads(T, F):
                a = _PyWrap._reorder_args(fn, obj, args, kwargs, M=1)
                if a is not None and _S.fn_can_call(fn, *a):
                    fn(*a)
                    return i32(0)
            _PyWrap._dispatch_error(F)
        else:
            F: Static[str] = "__new__"
            for fn in _S.fn_overloads(T, F):
                a = _PyWrap._reorder_args(fn, obj, args, kwargs, M=0)
                if a is not None and _S.fn_can_call(fn, *a):
                    x = fn(*a)
                    p = Ptr[PyObject](obj) + 1
                    Ptr[T](p.as_byte())[0] = x
                    return i32(0)
            _PyWrap._dispatch_error(F)

    def wrap_magic_call(obj: cobj, args: cobj, kwargs: cobj, T: type) -> cobj:
        F: Static[str] = "__call__"
        for fn in _S.fn_overloads(T, F):
            a = _PyWrap._reorder_args(fn, obj, args, kwargs, M=1)
            if a is not None and _S.fn_can_call(fn, *a):
                return fn(*a).__to_py__()
        _PyWrap._dispatch_error(F)

    def _wrap_cmp(obj: cobj, other: cobj, T: type, F: Static[str]) -> cobj:
        return _PyWrap._wrap(
            (obj, other), T=T, F=F,
            map=lambda f, a: f(*a).__to_py__()
        )

    def wrap_magic_lt(obj: cobj, other: cobj, T: type):
        return _PyWrap._wrap_cmp(obj, other, T, "__lt__")

    def wrap_magic_le(obj: cobj, other: cobj, T: type):
        return _PyWrap._wrap_cmp(obj, other, T, "__le__")

    def wrap_magic_eq(obj: cobj, other: cobj, T: type):
        return _PyWrap._wrap_cmp(obj, other, T, "__eq__")

    def wrap_magic_ne(obj: cobj, other: cobj, T: type):
        return _PyWrap._wrap_cmp(obj, other, T, "__ne__")

    def wrap_magic_gt(obj: cobj, other: cobj, T: type):
        return _PyWrap._wrap_cmp(obj, other, T, "__gt__")

    def wrap_magic_ge(obj: cobj, other: cobj, T: type):
        return _PyWrap._wrap_cmp(obj, other, T, "__ge__")

    def wrap_cmp(obj: cobj, other: cobj, op: i32, C: type) -> cobj:
        if hasattr(C, "__lt__") and op == 0i32:
            return _PyWrap.wrap_magic_lt(obj, other, C)
        elif hasattr(C, "__le__") and op == 1i32:
            return _PyWrap.wrap_magic_le(obj, other, C)
        elif hasattr(C, "__eq__") and op == 2i32:
            return _PyWrap.wrap_magic_eq(obj, other, C)
        elif hasattr(C, "__ne__") and op == 3i32:
            return _PyWrap.wrap_magic_ne(obj, other, C)
        elif hasattr(C, "__gt__") and op == 4i32:
            return _PyWrap.wrap_magic_gt(obj, other, C)
        elif hasattr(C, "__ge__") and op == 5i32:
            return _PyWrap.wrap_magic_ge(obj, other, C)
        else:
            Py_IncRef(Py_NotImplemented)
            return Py_NotImplemented

    def wrap_magic_getitem(obj: cobj, idx: cobj, T: type):
        return _PyWrap._wrap(
            (obj, idx), T=T, F="__getitem__",
            map=lambda f, a: f(*a).__to_py__()
        )

    def wrap_magic_setitem(obj: cobj, idx: cobj, val: cobj, T: type) -> i32:
        if val == cobj():
            _PyWrap._wrap(
                (obj, idx), T=T, F="__delitem__",
                map=lambda f, a: f(*a)
            )
        else:
            _PyWrap._wrap(
                (obj, idx, val), T=T, F="__setitem__",
                map=lambda f, a: f(*a)
            )
        return i32(0)

    class IterWrap:
        _gen: cobj
        T: type

        def _init(obj: cobj, T: type) -> cobj:
            return _PyWrap.IterWrap(T.__from_py__(obj)).__to_py__()

        @realize_without_self
        def __init__(self, it: T):
            self._gen = it.__iter__().__raw__()

        def _iter(obj: cobj) -> cobj:
            T  # need separate fn for each instantiation
            p = Ptr[PyObject](obj)
            o = p[0]
            p[0] = PyObject(o.refcnt + 1, o.pytype)
            return obj

        def _iternext(self: cobj) -> cobj:
            pt = _PyWrap.IterWrap[T].__from_py__(self)
            if pt._gen == cobj():
                return cobj()

            gt = type(T().__iter__())(pt._gen)
            if gt.done():
                pt._gen = cobj()
                return cobj()
            else:
                return gt.next().__to_py__()

        def __to_py__(self):
            return _PyWrap.wrap_to_py(self)

        def __from_py__(obj: cobj):
            return _PyWrap.wrap_from_py(obj, _PyWrap.IterWrap[T])

    def wrap_magic_iter(obj: cobj, T: type) -> cobj:
        return _PyWrap.IterWrap._init(obj, T)

    def wrap_multiple(
        obj: cobj, args: Ptr[cobj], nargs: int, _kwds: cobj, T: type, F: Static[str],
        M: Static[int] = 1
    ):
        kwds = Ptr[str]()
        nkw = 0
        if _kwds:
            nkw = PyTuple_Size(_kwds)
            kwds = Ptr[str](nkw)
            for i in range(nkw):
                kwds[i] = str.__from_py__(PyTuple_GetItem(_kwds, i))

        for fn in _S.fn_overloads(T, F):
            a = _PyWrap._reorder_args_fastcall(fn, obj, args, nargs, kwds, nkw, M)
            if a is not None and _S.fn_can_call(fn, *a):
                return fn(*a).__to_py__()

        _PyWrap._dispatch_error(F)

    def wrap_get(obj: cobj, closure: cobj, T: type, S: Static[str]):
        return getattr(T.__from_py__(obj), S).__to_py__()

    def wrap_set(obj: cobj, what: cobj, closure: cobj, T: type, S: Static[str]) -> i32:
        t = T.__from_py__(obj)
        val = type(getattr(t, S)).__from_py__(what)
        setattr(t, S, val)
        return i32(0)

    def py_type(T: type) -> cobj:
        return cobj()

    def wrap_to_py(o) -> cobj:
        O = type(o)
        P = PyWrapper[O]
        sz = sizeof(P)
        pytype = _PyWrap.py_type(O)
        mem = alloc_atomic_uncollectable(sz) if atomic(O) else alloc_uncollectable(sz)
        obj = Ptr[P](mem.as_byte())
        obj[0] = PyWrapper(PyObject(1, pytype), o)
        return obj.as_byte()

    def wrap_from_py(o: cobj, T: type) -> T:
        obj = Ptr[PyWrapper[T]](o)[0]
        pytype = _PyWrap.py_type(T)
        if obj.head.pytype != pytype:
            _conversion_error(T.__name__)
        return obj.data
