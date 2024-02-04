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

from internal.attributes import commutative
from internal.gc import alloc_atomic, free
from internal.types.complex import complex

@extend
class float:
    def __new__() -> float:
        return 0.0

    def __new__(what) -> float:
        # do not overload! (needed to avoid pyobj conversion)
        if isinstance(what, str) or isinstance(what, Optional[str]):
            return float._from_str(what)
        else:
            return what.__float__()

    def __repr__(self) -> str:
        return self.__format__("")

    def __copy__(self) -> float:
        return self

    def __deepcopy__(self) -> float:
        return self

    @pure
    @llvm
    def __int__(self) -> int:
        %0 = fptosi double %self to i64
        ret i64 %0

    def __float__(self) -> float:
        return self

    @pure
    @llvm
    def __bool__(self) -> bool:
        %0 = fcmp une double %self, 0.000000e+00
        %1 = zext i1 %0 to i8
        ret i8 %1

    def __complex__(self) -> complex:
        return complex(self, 0.0)

    def __pos__(self) -> float:
        return self

    @pure
    @llvm
    def __neg__(self) -> float:
        %0 = fneg double %self
        ret double %0

    @pure
    @commutative
    @llvm
    def __add__(a: float, b: float) -> float:
        %tmp = fadd double %a, %b
        ret double %tmp

    @pure
    @llvm
    def __sub__(a: float, b: float) -> float:
        %tmp = fsub double %a, %b
        ret double %tmp

    @pure
    @commutative
    @llvm
    def __mul__(a: float, b: float) -> float:
        %tmp = fmul double %a, %b
        ret double %tmp

    def __floordiv__(self, other: float) -> float:
        return self.__truediv__(other).__floor__()

    @pure
    @llvm
    def __truediv__(a: float, b: float) -> float:
        %tmp = fdiv double %a, %b
        ret double %tmp

    @pure
    @llvm
    def __mod__(a: float, b: float) -> float:
        %tmp = frem double %a, %b
        ret double %tmp

    def __divmod__(self, other: float) -> Tuple[float, float]:
        mod = self % other
        div = (self - mod) / other
        if mod:
            if (other < 0.0) != (mod < 0.0):
                mod += other
                div -= 1.0
        else:
            mod = (0.0).copysign(other)

        floordiv = 0.0
        if div:
            floordiv = div.__floor__()
            if div - floordiv > 0.5:
                floordiv += 1.0
        else:
            floordiv = (0.0).copysign(self / other)

        return (floordiv, mod)

    @pure
    @llvm
    def __eq__(a: float, b: float) -> bool:
        %tmp = fcmp oeq double %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ne__(a: float, b: float) -> bool:
        %tmp = fcmp une double %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __lt__(a: float, b: float) -> bool:
        %tmp = fcmp olt double %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __gt__(a: float, b: float) -> bool:
        %tmp = fcmp ogt double %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __le__(a: float, b: float) -> bool:
        %tmp = fcmp ole double %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ge__(a: float, b: float) -> bool:
        %tmp = fcmp oge double %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def sqrt(a: float) -> float:
        declare double @llvm.sqrt.f64(double %a)
        %tmp = call double @llvm.sqrt.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def sin(a: float) -> float:
        declare double @llvm.sin.f64(double %a)
        %tmp = call double @llvm.sin.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def cos(a: float) -> float:
        declare double @llvm.cos.f64(double %a)
        %tmp = call double @llvm.cos.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def exp(a: float) -> float:
        declare double @llvm.exp.f64(double %a)
        %tmp = call double @llvm.exp.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def exp2(a: float) -> float:
        declare double @llvm.exp2.f64(double %a)
        %tmp = call double @llvm.exp2.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def log(a: float) -> float:
        declare double @llvm.log.f64(double %a)
        %tmp = call double @llvm.log.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def log10(a: float) -> float:
        declare double @llvm.log10.f64(double %a)
        %tmp = call double @llvm.log10.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def log2(a: float) -> float:
        declare double @llvm.log2.f64(double %a)
        %tmp = call double @llvm.log2.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def __abs__(a: float) -> float:
        declare double @llvm.fabs.f64(double %a)
        %tmp = call double @llvm.fabs.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def __floor__(a: float) -> float:
        declare double @llvm.floor.f64(double %a)
        %tmp = call double @llvm.floor.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def __ceil__(a: float) -> float:
        declare double @llvm.ceil.f64(double %a)
        %tmp = call double @llvm.ceil.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def __trunc__(a: float) -> float:
        declare double @llvm.trunc.f64(double %a)
        %tmp = call double @llvm.trunc.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def rint(a: float) -> float:
        declare double @llvm.rint.f64(double %a)
        %tmp = call double @llvm.rint.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def nearbyint(a: float) -> float:
        declare double @llvm.nearbyint.f64(double %a)
        %tmp = call double @llvm.nearbyint.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def __round__(a: float) -> float:
        declare double @llvm.round.f64(double %a)
        %tmp = call double @llvm.round.f64(double %a)
        ret double %tmp

    @pure
    @llvm
    def __pow__(a: float, b: float) -> float:
        declare double @llvm.pow.f64(double %a, double %b)
        %tmp = call double @llvm.pow.f64(double %a, double %b)
        ret double %tmp

    @pure
    @llvm
    def min(a: float, b: float) -> float:
        declare double @llvm.minnum.f64(double %a, double %b)
        %tmp = call double @llvm.minnum.f64(double %a, double %b)
        ret double %tmp

    @pure
    @llvm
    def max(a: float, b: float) -> float:
        declare double @llvm.maxnum.f64(double %a, double %b)
        %tmp = call double @llvm.maxnum.f64(double %a, double %b)
        ret double %tmp

    @pure
    @llvm
    def copysign(a: float, b: float) -> float:
        declare double @llvm.copysign.f64(double %a, double %b)
        %tmp = call double @llvm.copysign.f64(double %a, double %b)
        ret double %tmp

    @pure
    @llvm
    def fma(a: float, b: float, c: float) -> float:
        declare double @llvm.fma.f64(double %a, double %b, double %c)
        %tmp = call double @llvm.fma.f64(double %a, double %b, double %c)
        ret double %tmp

    @nocapture
    @llvm
    def __atomic_xchg__(d: Ptr[float], b: float) -> None:
        %tmp = atomicrmw xchg ptr %d, double %b seq_cst
        ret {} {}

    @nocapture
    @llvm
    def __atomic_add__(d: Ptr[float], b: float) -> float:
        0:
        %1 = load atomic i64, ptr %d monotonic, align 8
        %2 = bitcast i64 %1 to double
        %3 = fadd double %2, %b
        %4 = bitcast double %3 to i64
        %5 = cmpxchg weak ptr %d, i64 %1, i64 %4 seq_cst monotonic, align 8
        %6 = extractvalue { i64, i1 } %5, 1
        br i1 %6, label %15, label %7
        7:                                                ; preds = %0, %7
        %8 = phi { i64, i1 } [ %13, %7 ], [ %5, %0 ]
        %9 = extractvalue { i64, i1 } %8, 0
        %10 = bitcast i64 %9 to double
        %11 = fadd double %10, %b
        %12 = bitcast double %11 to i64
        %13 = cmpxchg weak ptr %d, i64 %9, i64 %12 seq_cst monotonic, align 8
        %14 = extractvalue { i64, i1 } %13, 1
        br i1 %14, label %15, label %7
        15:                                               ; preds = %7, %0
        %16 = phi double [ %2, %0 ], [ %10, %7 ]
        ret double %16

    @nocapture
    @llvm
    def __atomic_sub__(d: Ptr[float], b: float) -> float:
        0:
        %1 = load atomic i64, ptr %d monotonic, align 8
        %2 = bitcast i64 %1 to double
        %3 = fsub double %2, %b
        %4 = bitcast double %3 to i64
        %5 = cmpxchg weak ptr %d, i64 %1, i64 %4 seq_cst monotonic, align 8
        %6 = extractvalue { i64, i1 } %5, 1
        br i1 %6, label %15, label %7
        7:                                                ; preds = %0, %7
        %8 = phi { i64, i1 } [ %13, %7 ], [ %5, %0 ]
        %9 = extractvalue { i64, i1 } %8, 0
        %10 = bitcast i64 %9 to double
        %11 = fsub double %10, %b
        %12 = bitcast double %11 to i64
        %13 = cmpxchg weak ptr %d, i64 %9, i64 %12 seq_cst monotonic, align 8
        %14 = extractvalue { i64, i1 } %13, 1
        br i1 %14, label %15, label %7
        15:                                               ; preds = %7, %0
        %16 = phi double [ %2, %0 ], [ %10, %7 ]
        ret double %16

    def __hash__(self) -> int:
        @nocapture
        @C
        def frexp(a: float, b: Ptr[Int[32]]) -> float: pass

        HASH_BITS = 61
        HASH_MODULUS = (1 << HASH_BITS) - 1
        HASH_INF = 314159
        HASH_NAN = 0
        INF = 1.0 / 0.0
        NAN = 0.0 / 0.0
        v = self

        if v == INF or v == -INF:
            return HASH_INF if v > 0 else -HASH_INF
        if v == NAN:
            return HASH_NAN

        _e = i32(0)
        m = frexp(v, __ptr__(_e))
        e = int(_e)

        sign = 1
        if m < 0:
            sign = -1
            m = -m

        x = 0
        while m:
            x = ((x << 28) & HASH_MODULUS) | x >> (HASH_BITS - 28)
            m *= 268435456.0  # 2**28
            e -= 28
            y = int(m)
            m -= y
            x += y
            if x >= HASH_MODULUS:
                x -= HASH_MODULUS

        e = e % HASH_BITS if e >= 0 else HASH_BITS - 1 - ((-1 - e) % HASH_BITS)
        x = ((x << e) & HASH_MODULUS) | x >> (HASH_BITS - e)

        x = x * sign
        if x == -1:
            x = -2
        return x

    def __match__(self, obj: float) -> bool:
        return self == obj

    @property
    def real(self) -> float:
        return self

    @property
    def imag(self) -> float:
        return 0.0

@extend
class float32:
    @pure
    @llvm
    def __new__(self: float) -> float32:
        %0 = fptrunc double %self to float
        ret float %0

    def __new__(what: float32) -> float32:
        return what

    def __new__(what: str) -> float32:
        return float32._from_str(what)

    def __new__() -> float32:
        return float32.__new__(0.0)

    def __repr__(self) -> str:
        return self.__float__().__repr__()

    def __format__(self, format_spec: str) -> str:
        return self.__float__().__format(format_spec)

    def __copy__(self) -> float32:
        return self

    def __deepcopy__(self) -> float32:
        return self

    @pure
    @llvm
    def __int__(self) -> int:
        %0 = fptosi float %self to i64
        ret i64 %0

    @pure
    @llvm
    def __float__(self) -> float:
        %0 = fpext float %self to double
        ret double %0

    @pure
    @llvm
    def __bool__(self) -> bool:
        %0 = fcmp une float %self, 0.000000e+00
        %1 = zext i1 %0 to i8
        ret i8 %1

    def __pos__(self) -> float32:
        return self

    @pure
    @llvm
    def __neg__(self) -> float32:
        %0 = fneg float %self
        ret float %0

    @pure
    @commutative
    @llvm
    def __add__(a: float32, b: float32) -> float32:
        %tmp = fadd float %a, %b
        ret float %tmp

    @pure
    @llvm
    def __sub__(a: float32, b: float32) -> float32:
        %tmp = fsub float %a, %b
        ret float %tmp

    @pure
    @commutative
    @llvm
    def __mul__(a: float32, b: float32) -> float32:
        %tmp = fmul float %a, %b
        ret float %tmp

    def __floordiv__(self, other: float32) -> float32:
        return self.__truediv__(other).__floor__()

    @pure
    @llvm
    def __truediv__(a: float32, b: float32) -> float32:
        %tmp = fdiv float %a, %b
        ret float %tmp

    @pure
    @llvm
    def __mod__(a: float32, b: float32) -> float32:
        %tmp = frem float %a, %b
        ret float %tmp

    def __divmod__(self, other: float32) -> Tuple[float32, float32]:
        mod = self % other
        div = (self - mod) / other
        if mod:
            if (other < float32(0.0)) != (mod < float32(0.0)):
                mod += other
                div -= float32(1.0)
        else:
            mod = float32(0.0).copysign(other)

        floordiv = float32(0.0)
        if div:
            floordiv = div.__floor__()
            if div - floordiv > float32(0.5):
                floordiv += float32(1.0)
        else:
            floordiv = float32(0.0).copysign(self / other)

        return (floordiv, mod)

    @pure
    @llvm
    def __eq__(a: float32, b: float32) -> bool:
        %tmp = fcmp oeq float %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ne__(a: float32, b: float32) -> bool:
        %tmp = fcmp une float %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __lt__(a: float32, b: float32) -> bool:
        %tmp = fcmp olt float %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __gt__(a: float32, b: float32) -> bool:
        %tmp = fcmp ogt float %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __le__(a: float32, b: float32) -> bool:
        %tmp = fcmp ole float %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ge__(a: float32, b: float32) -> bool:
        %tmp = fcmp oge float %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def sqrt(a: float32) -> float32:
        declare float @llvm.sqrt.f32(float %a)
        %tmp = call float @llvm.sqrt.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def sin(a: float32) -> float32:
        declare float @llvm.sin.f32(float %a)
        %tmp = call float @llvm.sin.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def cos(a: float32) -> float32:
        declare float @llvm.cos.f32(float %a)
        %tmp = call float @llvm.cos.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def exp(a: float32) -> float32:
        declare float @llvm.exp.f32(float %a)
        %tmp = call float @llvm.exp.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def exp2(a: float32) -> float32:
        declare float @llvm.exp2.f32(float %a)
        %tmp = call float @llvm.exp2.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def log(a: float32) -> float32:
        declare float @llvm.log.f32(float %a)
        %tmp = call float @llvm.log.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def log10(a: float32) -> float32:
        declare float @llvm.log10.f32(float %a)
        %tmp = call float @llvm.log10.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def log2(a: float32) -> float32:
        declare float @llvm.log2.f32(float %a)
        %tmp = call float @llvm.log2.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def __abs__(a: float32) -> float32:
        declare float @llvm.fabs.f32(float %a)
        %tmp = call float @llvm.fabs.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def __floor__(a: float32) -> float32:
        declare float @llvm.floor.f32(float %a)
        %tmp = call float @llvm.floor.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def __ceil__(a: float32) -> float32:
        declare float @llvm.ceil.f32(float %a)
        %tmp = call float @llvm.ceil.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def __trunc__(a: float32) -> float32:
        declare float @llvm.trunc.f32(float %a)
        %tmp = call float @llvm.trunc.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def rint(a: float32) -> float32:
        declare float @llvm.rint.f32(float %a)
        %tmp = call float @llvm.rint.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def nearbyint(a: float32) -> float32:
        declare float @llvm.nearbyint.f32(float %a)
        %tmp = call float @llvm.nearbyint.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def __round__(a: float32) -> float32:
        declare float @llvm.round.f32(float %a)
        %tmp = call float @llvm.round.f32(float %a)
        ret float %tmp

    @pure
    @llvm
    def __pow__(a: float32, b: float32) -> float32:
        declare float @llvm.pow.f32(float %a, float %b)
        %tmp = call float @llvm.pow.f32(float %a, float %b)
        ret float %tmp

    @pure
    @llvm
    def min(a: float32, b: float32) -> float32:
        declare float @llvm.minnum.f32(float %a, float %b)
        %tmp = call float @llvm.minnum.f32(float %a, float %b)
        ret float %tmp

    @pure
    @llvm
    def max(a: float32, b: float32) -> float32:
        declare float @llvm.maxnum.f32(float %a, float %b)
        %tmp = call float @llvm.maxnum.f32(float %a, float %b)
        ret float %tmp

    @pure
    @llvm
    def copysign(a: float32, b: float32) -> float32:
        declare float @llvm.copysign.f32(float %a, float %b)
        %tmp = call float @llvm.copysign.f32(float %a, float %b)
        ret float %tmp

    @pure
    @llvm
    def fma(a: float32, b: float32, c: float32) -> float32:
        declare float @llvm.fma.f32(float %a, float %b, float %c)
        %tmp = call float @llvm.fma.f32(float %a, float %b, float %c)
        ret float %tmp

    @nocapture
    @llvm
    def __atomic_xchg__(d: Ptr[float32], b: float32) -> None:
        %tmp = atomicrmw xchg ptr %d, float %b seq_cst
        ret {} {}

    @nocapture
    @llvm
    def __atomic_add__(d: Ptr[float32], b: float32) -> float32:
        0:
        %1 = load atomic i32, ptr %d monotonic, align 4
        %2 = bitcast i32 %1 to float
        %3 = fadd float %2, %b
        %4 = bitcast float %3 to i32
        %5 = cmpxchg weak ptr %d, i32 %1, i32 %4 seq_cst monotonic, align 4
        %6 = extractvalue { i32, i1 } %5, 1
        br i1 %6, label %15, label %7
        7:                                                ; preds = %0, %7
        %8 = phi { i32, i1 } [ %13, %7 ], [ %5, %0 ]
        %9 = extractvalue { i32, i1 } %8, 0
        %10 = bitcast i32 %9 to float
        %11 = fadd float %10, %b
        %12 = bitcast float %11 to i32
        %13 = cmpxchg weak ptr %d, i32 %9, i32 %12 seq_cst monotonic, align 4
        %14 = extractvalue { i32, i1 } %13, 1
        br i1 %14, label %15, label %7
        15:                                               ; preds = %7, %0
        %16 = phi float [ %2, %0 ], [ %10, %7 ]
        ret float %16

    @nocapture
    @llvm
    def __atomic_sub__(d: Ptr[float32], b: float32) -> float32:
        0:
        %1 = load atomic i32, ptr %d monotonic, align 4
        %2 = bitcast i32 %1 to float
        %3 = fsub float %2, %b
        %4 = bitcast float %3 to i32
        %5 = cmpxchg weak ptr %d, i32 %1, i32 %4 seq_cst monotonic, align 4
        %6 = extractvalue { i32, i1 } %5, 1
        br i1 %6, label %15, label %7
        7:                                                ; preds = %0, %7
        %8 = phi { i32, i1 } [ %13, %7 ], [ %5, %0 ]
        %9 = extractvalue { i32, i1 } %8, 0
        %10 = bitcast i32 %9 to float
        %11 = fsub float %10, %b
        %12 = bitcast float %11 to i32
        %13 = cmpxchg weak ptr %d, i32 %9, i32 %12 seq_cst monotonic, align 4
        %14 = extractvalue { i32, i1 } %13, 1
        br i1 %14, label %15, label %7
        15:                                               ; preds = %7, %0
        %16 = phi float [ %2, %0 ], [ %10, %7 ]
        ret float %16

    def __hash__(self) -> int:
        return self.__float__().__hash__()

    def __match__(self, obj: float32) -> bool:
        return self == obj

@extend
class float16:
    @pure
    @llvm
    def __new__(self: float) -> float16:
        %0 = fptrunc double %self to half
        ret half %0

    def __new__(what: float16) -> float16:
        return what

    def __new__(what: str) -> float16:
        return float16._from_str(what)

    def __new__() -> float16:
        return float16.__new__(0.0)

    def __repr__(self) -> str:
        return self.__float__().__repr__()

    def __format__(self, format_spec: str) -> str:
        return self.__float__().__format(format_spec)

    def __copy__(self) -> float16:
        return self

    def __deepcopy__(self) -> float16:
        return self

    @pure
    @llvm
    def __int__(self) -> int:
        %0 = fptosi half %self to i64
        ret i64 %0

    @pure
    @llvm
    def __float__(self) -> float:
        %0 = fpext half %self to double
        ret double %0

    @pure
    @llvm
    def __bool__(self) -> bool:
        %0 = fcmp une half %self, 0.000000e+00
        %1 = zext i1 %0 to i8
        ret i8 %1

    def __pos__(self) -> float16:
        return self

    @pure
    @llvm
    def __neg__(self) -> float16:
        %0 = fneg half %self
        ret half %0

    @pure
    @commutative
    @llvm
    def __add__(a: float16, b: float16) -> float16:
        %tmp = fadd half %a, %b
        ret half %tmp

    @pure
    @llvm
    def __sub__(a: float16, b: float16) -> float16:
        %tmp = fsub half %a, %b
        ret half %tmp

    @pure
    @commutative
    @llvm
    def __mul__(a: float16, b: float16) -> float16:
        %tmp = fmul half %a, %b
        ret half %tmp

    def __floordiv__(self, other: float16) -> float16:
        return self.__truediv__(other).__floor__()

    @pure
    @llvm
    def __truediv__(a: float16, b: float16) -> float16:
        %tmp = fdiv half %a, %b
        ret half %tmp

    @pure
    @llvm
    def __mod__(a: float16, b: float16) -> float16:
        %tmp = frem half %a, %b
        ret half %tmp

    def __divmod__(self, other: float16) -> Tuple[float16, float16]:
        mod = self % other
        div = (self - mod) / other
        if mod:
            if (other < float16(0.0)) != (mod < float16(0.0)):
                mod += other
                div -= float16(1.0)
        else:
            mod = float16(0.0).copysign(other)

        floordiv = float16(0.0)
        if div:
            floordiv = div.__floor__()
            if div - floordiv > float16(0.5):
                floordiv += float16(1.0)
        else:
            floordiv = float16(0.0).copysign(self / other)

        return (floordiv, mod)

    @pure
    @llvm
    def __eq__(a: float16, b: float16) -> bool:
        %tmp = fcmp oeq half %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ne__(a: float16, b: float16) -> bool:
        %tmp = fcmp une half %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __lt__(a: float16, b: float16) -> bool:
        %tmp = fcmp olt half %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __gt__(a: float16, b: float16) -> bool:
        %tmp = fcmp ogt half %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __le__(a: float16, b: float16) -> bool:
        %tmp = fcmp ole half %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ge__(a: float16, b: float16) -> bool:
        %tmp = fcmp oge half %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def sqrt(a: float16) -> float16:
        declare half @llvm.sqrt.f16(half %a)
        %tmp = call half @llvm.sqrt.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def sin(a: float16) -> float16:
        declare half @llvm.sin.f16(half %a)
        %tmp = call half @llvm.sin.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def cos(a: float16) -> float16:
        declare half @llvm.cos.f16(half %a)
        %tmp = call half @llvm.cos.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def exp(a: float16) -> float16:
        declare half @llvm.exp.f16(half %a)
        %tmp = call half @llvm.exp.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def exp2(a: float16) -> float16:
        declare half @llvm.exp2.f16(half %a)
        %tmp = call half @llvm.exp2.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def log(a: float16) -> float16:
        declare half @llvm.log.f16(half %a)
        %tmp = call half @llvm.log.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def log10(a: float16) -> float16:
        declare half @llvm.log10.f16(half %a)
        %tmp = call half @llvm.log10.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def log2(a: float16) -> float16:
        declare half @llvm.log2.f16(half %a)
        %tmp = call half @llvm.log2.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def __abs__(a: float16) -> float16:
        declare half @llvm.fabs.f16(half %a)
        %tmp = call half @llvm.fabs.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def __floor__(a: float16) -> float16:
        declare half @llvm.floor.f16(half %a)
        %tmp = call half @llvm.floor.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def __ceil__(a: float16) -> float16:
        declare half @llvm.ceil.f16(half %a)
        %tmp = call half @llvm.ceil.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def __trunc__(a: float16) -> float16:
        declare half @llvm.trunc.f16(half %a)
        %tmp = call half @llvm.trunc.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def rint(a: float16) -> float16:
        declare half @llvm.rint.f16(half %a)
        %tmp = call half @llvm.rint.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def nearbyint(a: float16) -> float16:
        declare half @llvm.nearbyint.f16(half %a)
        %tmp = call half @llvm.nearbyint.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def __round__(a: float16) -> float16:
        declare half @llvm.round.f16(half %a)
        %tmp = call half @llvm.round.f16(half %a)
        ret half %tmp

    @pure
    @llvm
    def __pow__(a: float16, b: float16) -> float16:
        declare half @llvm.pow.f16(half %a, half %b)
        %tmp = call half @llvm.pow.f16(half %a, half %b)
        ret half %tmp

    @pure
    @llvm
    def min(a: float16, b: float16) -> float16:
        declare half @llvm.minnum.f16(half %a, half %b)
        %tmp = call half @llvm.minnum.f16(half %a, half %b)
        ret half %tmp

    @pure
    @llvm
    def max(a: float16, b: float16) -> float16:
        declare half @llvm.maxnum.f16(half %a, half %b)
        %tmp = call half @llvm.maxnum.f16(half %a, half %b)
        ret half %tmp

    @pure
    @llvm
    def copysign(a: float16, b: float16) -> float16:
        declare half @llvm.copysign.f16(half %a, half %b)
        %tmp = call half @llvm.copysign.f16(half %a, half %b)
        ret half %tmp

    @pure
    @llvm
    def fma(a: float16, b: float16, c: float16) -> float16:
        declare half @llvm.fma.f16(half %a, half %b, half %c)
        %tmp = call half @llvm.fma.f16(half %a, half %b, half %c)
        ret half %tmp

    def __hash__(self) -> int:
        return self.__float__().__hash__()

    def __match__(self, obj: float16) -> bool:
        return self == obj

@extend
class bfloat16:
    @pure
    @llvm
    def __new__(self: float) -> bfloat16:
        %0 = fptrunc double %self to bfloat
        ret bfloat %0

    def __new__(what: bfloat16) -> bfloat16:
        return what

    def __new__(what: str) -> bfloat16:
        return bfloat16._from_str(what)

    def __new__() -> bfloat16:
        return bfloat16.__new__(0.0)

    def __repr__(self) -> str:
        return self.__float__().__repr__()

    def __format__(self, format_spec: str) -> str:
        return self.__float__().__format(format_spec)

    def __copy__(self) -> bfloat16:
        return self

    def __deepcopy__(self) -> bfloat16:
        return self

    @pure
    @llvm
    def __int__(self) -> int:
        %0 = fptosi bfloat %self to i64
        ret i64 %0

    @pure
    @llvm
    def __float__(self) -> float:
        %0 = fpext bfloat %self to double
        ret double %0

    @pure
    @llvm
    def __bool__(self) -> bool:
        %0 = fcmp une bfloat %self, 0.000000e+00
        %1 = zext i1 %0 to i8
        ret i8 %1

    def __pos__(self) -> bfloat16:
        return self

    @pure
    @llvm
    def __neg__(self) -> bfloat16:
        %0 = fneg bfloat %self
        ret bfloat %0

    @pure
    @commutative
    @llvm
    def __add__(a: bfloat16, b: bfloat16) -> bfloat16:
        %tmp = fadd bfloat %a, %b
        ret bfloat %tmp

    @pure
    @llvm
    def __sub__(a: bfloat16, b: bfloat16) -> bfloat16:
        %tmp = fsub bfloat %a, %b
        ret bfloat %tmp

    @pure
    @commutative
    @llvm
    def __mul__(a: bfloat16, b: bfloat16) -> bfloat16:
        %tmp = fmul bfloat %a, %b
        ret bfloat %tmp

    def __floordiv__(self, other: bfloat16) -> bfloat16:
        return self.__truediv__(other).__floor__()

    @pure
    @llvm
    def __truediv__(a: bfloat16, b: bfloat16) -> bfloat16:
        %tmp = fdiv bfloat %a, %b
        ret bfloat %tmp

    @pure
    @llvm
    def __mod__(a: bfloat16, b: bfloat16) -> bfloat16:
        %tmp = frem bfloat %a, %b
        ret bfloat %tmp

    def __divmod__(self, other: bfloat16) -> Tuple[bfloat16, bfloat16]:
        mod = self % other
        div = (self - mod) / other
        if mod:
            if (other < bfloat16(0.0)) != (mod < bfloat16(0.0)):
                mod += other
                div -= bfloat16(1.0)
        else:
            mod = bfloat16(0.0).copysign(other)

        floordiv = bfloat16(0.0)
        if div:
            floordiv = div.__floor__()
            if div - floordiv > bfloat16(0.5):
                floordiv += bfloat16(1.0)
        else:
            floordiv = bfloat16(0.0).copysign(self / other)

        return (floordiv, mod)

    @pure
    @llvm
    def __eq__(a: bfloat16, b: bfloat16) -> bool:
        %tmp = fcmp oeq bfloat %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ne__(a: bfloat16, b: bfloat16) -> bool:
        %tmp = fcmp une bfloat %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __lt__(a: bfloat16, b: bfloat16) -> bool:
        %tmp = fcmp olt bfloat %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __gt__(a: bfloat16, b: bfloat16) -> bool:
        %tmp = fcmp ogt bfloat %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __le__(a: bfloat16, b: bfloat16) -> bool:
        %tmp = fcmp ole bfloat %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ge__(a: bfloat16, b: bfloat16) -> bool:
        %tmp = fcmp oge bfloat %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def sqrt(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.sqrt.bf16(bfloat %a)
        %tmp = call bfloat @llvm.sqrt.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def sin(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.sin.bf16(bfloat %a)
        %tmp = call bfloat @llvm.sin.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def cos(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.cos.bf16(bfloat %a)
        %tmp = call bfloat @llvm.cos.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def exp(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.exp.bf16(bfloat %a)
        %tmp = call bfloat @llvm.exp.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def exp2(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.exp2.bf16(bfloat %a)
        %tmp = call bfloat @llvm.exp2.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def log(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.log.bf16(bfloat %a)
        %tmp = call bfloat @llvm.log.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def log10(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.log10.bf16(bfloat %a)
        %tmp = call bfloat @llvm.log10.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def log2(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.log2.bf16(bfloat %a)
        %tmp = call bfloat @llvm.log2.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def __abs__(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.fabs.bf16(bfloat %a)
        %tmp = call bfloat @llvm.fabs.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def __floor__(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.floor.bf16(bfloat %a)
        %tmp = call bfloat @llvm.floor.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def __ceil__(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.ceil.bf16(bfloat %a)
        %tmp = call bfloat @llvm.ceil.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def __trunc__(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.trunc.bf16(bfloat %a)
        %tmp = call bfloat @llvm.trunc.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def rint(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.rint.bf16(bfloat %a)
        %tmp = call bfloat @llvm.rint.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def nearbyint(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.nearbyint.bf16(bfloat %a)
        %tmp = call bfloat @llvm.nearbyint.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def __round__(a: bfloat16) -> bfloat16:
        declare bfloat @llvm.round.bf16(bfloat %a)
        %tmp = call bfloat @llvm.round.bf16(bfloat %a)
        ret bfloat %tmp

    @pure
    @llvm
    def __pow__(a: bfloat16, b: bfloat16) -> bfloat16:
        declare bfloat @llvm.pow.bf16(bfloat %a, bfloat %b)
        %tmp = call bfloat @llvm.pow.bf16(bfloat %a, bfloat %b)
        ret bfloat %tmp

    @pure
    @llvm
    def min(a: bfloat16, b: bfloat16) -> bfloat16:
        declare bfloat @llvm.minnum.bf16(bfloat %a, bfloat %b)
        %tmp = call bfloat @llvm.minnum.bf16(bfloat %a, bfloat %b)
        ret bfloat %tmp

    @pure
    @llvm
    def max(a: bfloat16, b: bfloat16) -> bfloat16:
        declare bfloat @llvm.maxnum.bf16(bfloat %a, bfloat %b)
        %tmp = call bfloat @llvm.maxnum.bf16(bfloat %a, bfloat %b)
        ret bfloat %tmp

    @pure
    @llvm
    def copysign(a: bfloat16, b: bfloat16) -> bfloat16:
        declare bfloat @llvm.copysign.bf16(bfloat %a, bfloat %b)
        %tmp = call bfloat @llvm.copysign.bf16(bfloat %a, bfloat %b)
        ret bfloat %tmp

    @pure
    @llvm
    def fma(a: bfloat16, b: bfloat16, c: bfloat16) -> bfloat16:
        declare bfloat @llvm.fma.bf16(bfloat %a, bfloat %b, bfloat %c)
        %tmp = call bfloat @llvm.fma.bf16(bfloat %a, bfloat %b, bfloat %c)
        ret bfloat %tmp

    def __hash__(self) -> int:
        return self.__float__().__hash__()

    def __match__(self, obj: bfloat16) -> bool:
        return self == obj

@extend
class float128:
    @pure
    @llvm
    def __new__(self: float) -> float128:
        %0 = fpext double %self to fp128
        ret fp128 %0

    def __new__(what: float128) -> float128:
        return what

    def __new__() -> float128:
        return float128.__new__(0.0)

    def __repr__(self) -> str:
        return self.__float__().__repr__()

    def __format__(self, format_spec: str) -> str:
        return self.__float__().__format(format_spec)

    def __copy__(self) -> float128:
        return self

    def __deepcopy__(self) -> float128:
        return self

    @pure
    @llvm
    def __int__(self) -> int:
        %0 = fptosi fp128 %self to i64
        ret i64 %0

    @pure
    @llvm
    def __float__(self) -> float:
        %0 = fptrunc fp128 %self to double
        ret double %0

    @pure
    @llvm
    def __bool__(self) -> bool:
        %0 = fcmp une fp128 %self, 0xL00000000000000000000000000000000
        %1 = zext i1 %0 to i8
        ret i8 %1

    def __pos__(self) -> float128:
        return self

    @pure
    @llvm
    def __neg__(self) -> float128:
        %0 = fneg fp128 %self
        ret fp128 %0

    @pure
    @commutative
    @llvm
    def __add__(a: float128, b: float128) -> float128:
        %tmp = fadd fp128 %a, %b
        ret fp128 %tmp

    @pure
    @llvm
    def __sub__(a: float128, b: float128) -> float128:
        %tmp = fsub fp128 %a, %b
        ret fp128 %tmp

    @pure
    @commutative
    @llvm
    def __mul__(a: float128, b: float128) -> float128:
        %tmp = fmul fp128 %a, %b
        ret fp128 %tmp

    def __floordiv__(self, other: float128) -> float128:
        return self.__truediv__(other).__floor__()

    @pure
    @llvm
    def __truediv__(a: float128, b: float128) -> float128:
        %tmp = fdiv fp128 %a, %b
        ret fp128 %tmp

    @pure
    @llvm
    def __mod__(a: float128, b: float128) -> float128:
        %tmp = frem fp128 %a, %b
        ret fp128 %tmp

    def __divmod__(self, other: float128) -> Tuple[float128, float128]:
        mod = self % other
        div = (self - mod) / other
        if mod:
            if (other < float128(0.0)) != (mod < float128(0)):
                mod += other
                div -= float128(1.0)
        else:
            mod = float128(0.0).copysign(other)

        floordiv = float128(0.0)
        if div:
            floordiv = div.__floor__()
            if div - floordiv > float128(0.5):
                floordiv += float128(1.0)
        else:
            floordiv = float128(0.0).copysign(self / other)

        return (floordiv, mod)

    @pure
    @llvm
    def __eq__(a: float128, b: float128) -> bool:
        %tmp = fcmp oeq fp128 %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ne__(a: float128, b: float128) -> bool:
        %tmp = fcmp une fp128 %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __lt__(a: float128, b: float128) -> bool:
        %tmp = fcmp olt fp128 %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __gt__(a: float128, b: float128) -> bool:
        %tmp = fcmp ogt fp128 %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __le__(a: float128, b: float128) -> bool:
        %tmp = fcmp ole fp128 %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def __ge__(a: float128, b: float128) -> bool:
        %tmp = fcmp oge fp128 %a, %b
        %res = zext i1 %tmp to i8
        ret i8 %res

    @pure
    @llvm
    def sqrt(a: float128) -> float128:
        declare fp128 @llvm.sqrt.f128(fp128 %a)
        %tmp = call fp128 @llvm.sqrt.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def sin(a: float128) -> float128:
        declare fp128 @llvm.sin.f128(fp128 %a)
        %tmp = call fp128 @llvm.sin.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def cos(a: float128) -> float128:
        declare fp128 @llvm.cos.f128(fp128 %a)
        %tmp = call fp128 @llvm.cos.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def exp(a: float128) -> float128:
        declare fp128 @llvm.exp.f128(fp128 %a)
        %tmp = call fp128 @llvm.exp.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def exp2(a: float128) -> float128:
        declare fp128 @llvm.exp2.f128(fp128 %a)
        %tmp = call fp128 @llvm.exp2.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def log(a: float128) -> float128:
        declare fp128 @llvm.log.f128(fp128 %a)
        %tmp = call fp128 @llvm.log.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def log10(a: float128) -> float128:
        declare fp128 @llvm.log10.f128(fp128 %a)
        %tmp = call fp128 @llvm.log10.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def log2(a: float128) -> float128:
        declare fp128 @llvm.log2.f128(fp128 %a)
        %tmp = call fp128 @llvm.log2.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def __abs__(a: float128) -> float128:
        declare fp128 @llvm.fabs.f128(fp128 %a)
        %tmp = call fp128 @llvm.fabs.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def __floor__(a: float128) -> float128:
        declare fp128 @llvm.floor.f128(fp128 %a)
        %tmp = call fp128 @llvm.floor.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def __ceil__(a: float128) -> float128:
        declare fp128 @llvm.ceil.f128(fp128 %a)
        %tmp = call fp128 @llvm.ceil.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def __trunc__(a: float128) -> float128:
        declare fp128 @llvm.trunc.f128(fp128 %a)
        %tmp = call fp128 @llvm.trunc.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def rint(a: float128) -> float128:
        declare fp128 @llvm.rint.f128(fp128 %a)
        %tmp = call fp128 @llvm.rint.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def nearbyint(a: float128) -> float128:
        declare fp128 @llvm.nearbyint.f128(fp128 %a)
        %tmp = call fp128 @llvm.nearbyint.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def __round__(a: float128) -> float128:
        declare fp128 @llvm.round.f128(fp128 %a)
        %tmp = call fp128 @llvm.round.f128(fp128 %a)
        ret fp128 %tmp

    @pure
    @llvm
    def __pow__(a: float128, b: float128) -> float128:
        declare fp128 @llvm.pow.f128(fp128 %a, fp128 %b)
        %tmp = call fp128 @llvm.pow.f128(fp128 %a, fp128 %b)
        ret fp128 %tmp

    @pure
    @llvm
    def min(a: float128, b: float128) -> float128:
        declare fp128 @llvm.minnum.f128(fp128 %a, fp128 %b)
        %tmp = call fp128 @llvm.minnum.f128(fp128 %a, fp128 %b)
        ret fp128 %tmp

    @pure
    @llvm
    def max(a: float128, b: float128) -> float128:
        declare fp128 @llvm.maxnum.f128(fp128 %a, fp128 %b)
        %tmp = call fp128 @llvm.maxnum.f128(fp128 %a, fp128 %b)
        ret fp128 %tmp

    @pure
    @llvm
    def copysign(a: float128, b: float128) -> float128:
        declare fp128 @llvm.copysign.f128(fp128 %a, fp128 %b)
        %tmp = call fp128 @llvm.copysign.f128(fp128 %a, fp128 %b)
        ret fp128 %tmp

    @pure
    @llvm
    def fma(a: float128, b: float128, c: float128) -> float128:
        declare fp128 @llvm.fma.f128(fp128 %a, fp128 %b, fp128 %c)
        %tmp = call fp128 @llvm.fma.f128(fp128 %a, fp128 %b, fp128 %c)
        ret fp128 %tmp

    def __hash__(self) -> int:
        return self.__float__().__hash__()

    def __match__(self, obj: float128) -> bool:
        return self == obj

@extend
class float:
    def __suffix_f32__(double) -> float32:
        return float32.__new__(double)

    def __suffix_f16__(double) -> float16:
        return float16.__new__(double)

    def __suffix_bf16__(double) -> bfloat16:
        return bfloat16.__new__(double)

    def __suffix_f128__(double) -> float128:
        return float128.__new__(double)

f16 = float16
bf16 = bfloat16
f32 = float32
f64 = float
f128 = float128
