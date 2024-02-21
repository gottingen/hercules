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

class object:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object at {self.__raw__()}>"

def id(x) -> int:
    if isinstance(x, ByRef):
        return int(x.__raw__())
    else:
        return 0

_stdout = _C.seq_stdout()

def print(*args, sep: str = " ", end: str = "\n", file=_stdout, flush: bool = False):
    """
    Print args to the text stream file.
    """
    fp = cobj()
    if isinstance(file, cobj):
        fp = file
    else:
        fp = file.fp
    i = 0
    for a in args:
        if i and sep:
            _C.seq_print_full(sep, fp)
        _C.seq_print_full(str(a), fp)
        i += 1
    _C.seq_print_full(end, fp)
    if flush:
        _C.fflush(fp)

@extend
class __internal__:
    def print(*args):
        print(*args, flush=True, file=_C.seq_stdout())

def min(*args, key=None, default=None):
    if staticlen(args) == 0:
        compile_error("min() expected at least 1 argument, got 0")
    elif staticlen(args) > 1 and default is not None:
        compile_error("min() 'default' argument only allowed for iterables")
    elif staticlen(args) == 1:
        x = args[0].__iter__()
        if not x.done():
            s = x.next()
            while not x.done():
                i = x.next()
                if key is None:
                    if i < s:
                        s = i
                else:
                    if key(i) < key(s):
                        s = i
            x.destroy()
            return s
        else:
            x.destroy()
        if default is None:
            raise ValueError("min() arg is an empty sequence")
        else:
            return default
    elif staticlen(args) == 2:
        a, b = args
        if key is None:
            return a if a <= b else b
        else:
            return a if key(a) <= key(b) else b
    else:
        m = args[0]
        for i in args[1:]:
            if key is None:
                if i < m:
                    m = i
            else:
                if key(i) < key(m):
                    m = i
        return m

def max(*args, key=None, default=None):
    if staticlen(args) == 0:
        compile_error("max() expected at least 1 argument, got 0")
    elif staticlen(args) > 1 and default is not None:
        compile_error("max() 'default' argument only allowed for iterables")
    elif staticlen(args) == 1:
        x = args[0].__iter__()
        if not x.done():
            s = x.next()
            while not x.done():
                i = x.next()
                if key is None:
                    if i > s:
                        s = i
                else:
                    if key(i) > key(s):
                        s = i
            x.destroy()
            return s
        else:
            x.destroy()
        if default is None:
            raise ValueError("max() arg is an empty sequence")
        else:
            return default
    elif staticlen(args) == 2:
        a, b = args
        if key is None:
            return a if a >= b else b
        else:
            return a if key(a) >= key(b) else b
    else:
        m = args[0]
        for i in args[1:]:
            if key is None:
                if i > m:
                    m = i
            else:
                if key(i) > key(m):
                    m = i
        return m

def len(x) -> int:
    """
    Return the length of x
    """
    return x.__len__()

def iter(x):
    """
    Return an iterator for the given object
    """
    return x.__iter__()

def abs(x):
    """
    Return the absolute value of x
    """
    return x.__abs__()

def hash(x) -> int:
    """
    Returns hashed value only for immutable objects
    """
    return x.__hash__()

def ord(s: str) -> int:
    """
    Return an integer representing the Unicode code point of s
    """
    if len(s) != 1:
        raise TypeError(
            f"ord() expected a character, but string of length {len(s)} found"
        )
    return int(s.ptr[0])

def divmod(a, b):
    if hasattr(a, "__divmod__"):
        return a.__divmod__(b)
    else:
        return (a // b, a % b)

def chr(i: int) -> str:
    """
    Return a string representing a character whose Unicode
    code point is an integer
    """
    p = cobj(1)
    p[0] = byte(i)
    return str(p, 1)

def next(g: Generator[T], default: Optional[T] = None, T: type) -> T:
    """
    Return the next item from g
    """
    if g.done():
        if default is not None:
            return default.__val__()
        else:
            raise StopIteration()
    return g.next()

def any(x: Generator[T], T: type) -> bool:
    """
    Returns True if any item in x is true,
    False otherwise
    """
    for a in x:
        if a:
            return True
    return False

def all(x: Generator[T], T: type) -> bool:
    """
    Returns True when all elements in x are true,
    False otherwise
    """
    for a in x:
        if not a:
            return False
    return True

def zip(*args):
    """
    Returns a zip object, which is an iterator of tuples
    that aggregates elements based on the iterables passed
    """
    if staticlen(args) == 0:
        yield from List[int]()
    else:
        iters = tuple(iter(i) for i in args)
        done = False
        while not done:
            for i in iters:
                if i.done():
                    done = True
            if not done:
                yield tuple(i.next() for i in iters)
        for i in iters:
            i.destroy()

def filter(f: Callable[[T], bool], x: Generator[T], T: type) -> Generator[T]:
    """
    Returns all a from the iterable x that are filtered by f
    """
    for a in x:
        if f(a):
            yield a

def map(f, *args):
    """
    Applies a function on all a in x and returns map object
    """
    if staticlen(args) == 0:
        compile_error("map() expects at least one iterator")
    elif staticlen(args) == 1:
        for a in args[0]:
            yield f(a)
    else:
        for a in zip(*args):
            yield f(*a)

def enumerate(x, start: int = 0):
    """
    Creates a tuple containing a count (from start which defaults
    to 0) and the values obtained from iterating over x
    """
    i = start
    for a in x:
        yield (i, a)
        i += 1

def staticenumerate(tup):
    i = -1
    return tuple(((i := i + 1), t) for t in tup)
    i

def echo(x):
    """
    Print and return argument
    """
    print x
    return x

def reversed(x):
    """
    Return an iterator that accesses x in the reverse order
    """
    if hasattr(x, "__reversed__"):
        return x.__reversed__()
    else:
        i = x.__len__() - 1
        while i >= 0:
            yield x[i]
            i -= 1

def round(x, n=0):
    """
    Return the x rounded off to the given
    n digits after the decimal point.
    """
    nx = float.__pow__(10.0, n)
    return float.__round__(x * nx) / nx

def _sum_start(x, start):
    if isinstance(x.__iter__(), Generator[float]) and isinstance(start, int):
        return float(start)
    else:
        return start

def sum(x, start=0):
    """
    Return the sum of the items added together from x
    """
    s = _sum_start(x, start)

    for a in x:
        # don't use += to avoid calling iadd
        if isinstance(a, bool):
            s = s + (1 if a else 0)
        else:
            s = s + a

    return s

def repr(x):
    """Return the string representation of x"""
    return x.__repr__()

def _int_format(a: int, base: int, prefix: str = ""):
    assert base == 2 or base == 8 or base == 10 or base == 16
    chars = "0123456789abcdef-"

    b = a
    digits = 0
    while b != 0:
        digits += 1
        b //= base

    sz = digits + (1 if a <= 0 else 0) + len(prefix)
    p = Ptr[byte](sz)
    q = p

    if a < 0:
        q[0] = chars[-1].ptr[0]
        q += 1

    if prefix:
        str.memcpy(q, prefix.ptr, len(prefix))
        q += len(prefix)

    if digits != 0:
        b = a
        q += digits - 1
        i = 1
        while b != 0:
            i += 1
            q[0] = chars.ptr[abs(b % base)]
            q += -1
            b //= base
    else:
        q[0] = chars.ptr[0]

    return str(p, sz)

def bin(n):
    return _int_format(n.__index__(), 2, "0b")

def oct(n):
    return _int_format(n.__index__(), 8, "0o")

def hex(n):
    return _int_format(n.__index__(), 16, "0x")

def pow(base: float, exp: float):
    return base ** exp

@overload
def pow(base: int, exp: int, mod: Optional[int] = None):
    if exp < 0:
        raise ValueError("pow() negative int exponent not supported")

    if mod is not None:
        if mod == 0:
            raise ValueError("pow() 3rd argument cannot be 0")
        base %= mod

    result = 1
    while exp > 0:
        if exp & 1:
            x = result * base
            result = x % mod if mod is not None else x
        y = base * base
        base = y % mod if mod is not None else y
        exp >>= 1
    return result % mod if mod is not None else result

@extend
class int:
    def _from_str(s: str, base: int):
        from internal.gc import alloc_atomic, free

        if base < 0 or base > 36 or base == 1:
            raise ValueError("int() base must be >= 2 and <= 36, or 0")

        s0 = s
        s = s.strip()
        buf = __array__[byte](32)
        n = len(s)
        need_dyn_alloc = n >= len(buf)

        p = alloc_atomic(n + 1) if need_dyn_alloc else buf.ptr
        str.memcpy(p, s.ptr, n)
        p[n] = byte(0)

        end = cobj()
        result = _C.strtoll(p, __ptr__(end), i32(base))

        if need_dyn_alloc:
            free(p)

        if n == 0 or end != p + n:
            raise ValueError(
                f"invalid literal for int() with base {base}: {s0.__repr__()}"
            )

        return result

@extend
class float:
    def _from_str(s: str) -> float:
        s0 = s
        s = s.strip()
        buf = __array__[byte](32)
        n = len(s)
        need_dyn_alloc = n >= len(buf)

        p = alloc_atomic(n + 1) if need_dyn_alloc else buf.ptr
        str.memcpy(p, s.ptr, n)
        p[n] = byte(0)

        end = cobj()
        result = _C.strtod(p, __ptr__(end))

        if need_dyn_alloc:
            free(p)

        if n == 0 or end != p + n:
            raise ValueError(f"could not convert string to float: {s0.__repr__()}")

        return result

@extend
class complex:
    def _from_str(v: str) -> complex:
        def parse_error():
            raise ValueError("complex() arg is a malformed string")

        buf = __array__[byte](32)
        n = len(v)
        need_dyn_alloc = False
        if n >= len(buf):
            need_dyn_alloc = True

        #need_dyn_alloc = n >= len(buf)

        s = buf.ptr
        ss = cobj()
        if need_dyn_alloc:
            s = alloc_atomic(n + 1)
            ss = s
        ##s = alloc_atomic(n + 1) if need_dyn_alloc else buf.ptr
        str.memcpy(s, v.ptr, n)
        s[n] = byte(0)

        x = 0.0
        y = 0.0
        z = 0.0
        got_bracket = False
        start = s
        end = cobj()

        while str._isspace(s[0]):
            s += 1

        if s[0] == byte(40):  # '('
            got_bracket = True
            s += 1
            while str._isspace(s[0]):
                s += 1

        z = _C.strtod(s, __ptr__(end))

        if end != s:
            s = end

            if s[0] == byte(43) or s[0] == byte(45):  # '+' '-'
                x = z
                y = _C.strtod(s, __ptr__(end))

                if end != s:
                    s = end
                else:
                    y = 1.0 if s[0] == byte(43) else -1.0
                    s += 1

                if not (s[0] == byte(106) or s[0] == byte(74)):  # 'j' 'J'
                    if need_dyn_alloc:
                        free(ss)
                    parse_error()

                s += 1
            elif s[0] == byte(106) or s[0] == byte(74):  # 'j' 'J'
                s += 1
                y = z
            else:
                x = z
        else:
            if s[0] == byte(43) or s[0] == byte(45):  # '+' '-'
                y = 1.0 if s[0] == byte(43) else -1.0
                s += 1
            else:
                y = 1.0

            if not (s[0] == byte(106) or s[0] == byte(74)):  # 'j' 'J'
                if need_dyn_alloc:
                    free(ss)
                parse_error()

            s += 1

        while str._isspace(s[0]):
            s += 1

        if got_bracket:
            if s[0] != byte(41):  # ')'
                if need_dyn_alloc:
                    free(ss)
                parse_error()
            s += 1
            while str._isspace(s[0]):
                s += 1

        if s - start != n:
            if need_dyn_alloc:
                free(ss)
            parse_error()

        if need_dyn_alloc:
            free(ss)
        return complex(x, y)

@extend
class float32:
    def _from_str(s: str) -> float32:
        return float32(float._from_str(s))

@extend
class float16:
    def _from_str(s: str) -> float16:
        return float16(float._from_str(s))

@extend
class bfloat16:
    def _from_str(s: str) -> bfloat16:
        return bfloat16(float._from_str(s))

@extend
class complex64:
    def _from_str(s: str) -> complex64:
        return complex64(complex._from_str(s))

def _jit_display(x, s: Static[str], bundle: Set[str] = Set[str]()):
    if isinstance(x, None):
        return
    if hasattr(x, "_repr_mimebundle_") and s == "jupyter":
        d = x._repr_mimebundle_(bundle)
        # TODO: pick appropriate mime
        mime = next(d.keys()) # just pick first
        print(f"\x00\x00__hercules/mime__\x00{mime}\x00{d[mime]}", end='')
    elif hasattr(x, "__repr__"):
        print(x.__repr__(), end='')
    elif hasattr(x, "__str__"):
        print(x.__str__(), end='')
