#%% pass,barebones
pass

#%% continue_error,barebones
continue #! 'continue' outside loop

#%% break_error,barebones
break #! 'break' outside loop

#%% assign,barebones
a = 1
print a #: 1
a = 2
print a #: 2

x, y = 1, 2
print x, y #: 1 2
(x, y) = (3, 4)
print x, y #: 3 4
x, y = (1, 2)
print x, y #: 1 2
(x, y) = 3, 4
print x, y #: 3 4
(x, y) = [3, 4]
print x, y #: 3 4
[x, y] = [1, 2]
print x, y #: 1 2
[x, y] = (4, 3)
print x, y #: 4 3

l = list(iter(range(10)))
[a, b, *lx, c, d] = l
print a, b, lx, c, d #: 0 1 [2, 3, 4, 5, 6, 7] 8 9
a, b, *lx = l
print a, b, lx #: 0 1 [2, 3, 4, 5, 6, 7, 8, 9]
*lx, a, b = l
print lx, a, b #: [0, 1, 2, 3, 4, 5, 6, 7] 8 9
*xz, a, b = (1, 2, 3, 4, 5)
print xz, a, b #: (1, 2, 3) 4 5
(*ex,) = [1, 2, 3]
print ex #: [1, 2, 3]

#%% assign_str,barebones
sa, sb = 'XY'
print sa, sb #: X Y
(sa, sb), sc = 'XY', 'Z'
print sa, sb, sc #: X Y Z
sa, *la = 'X'
print sa, la, 1 #: X  1
sa, *la = 'XYZ'
print sa, la #: X YZ
(xa,xb), *xc, xd = [1,2],'this'
print xa, xb, xc, xd #: 1 2 () this
(a, b), (sc, *sl) = [1,2], 'this'
print a, b, sc, sl #: 1 2 t his

#%% assign_index_dot,barebones
class Foo:
    a: int
    def __setitem__(self, i: int, t: int):
        self.a += i * t
f = Foo()
f.a = 5
print f.a #: 5
f[3] = 5
print f.a #: 20
f[1] = -8
print f.a #: 12


def foo():
    print('foo')
    return 0
v = [0]
v[foo()] += 1
#: foo
print(v)
#: [1]

#%% assign_err_1,barebones
a, *b, c, *d = 1,2,3,4,5 #! multiple starred expressions in assignment

#%% assign_err_2,barebones
a = [1, 2, 3]
a[1]: int = 3 #! syntax error, unexpected ':'

#%% assign_err_3,barebones
a = 5
a.x: int = 3 #! syntax error, unexpected ':'

#%% assign_err_4,barebones
*x = range(5) #! cannot assign to given expression

#%% assign_err_5,barebones
try:
    (sa, sb), sc = 'XYZ'
except IndexError:
    print "assign failed" #: assign failed

#%% assign_comprehension,barebones
g = ((b, a, c) for a, *b, c in ['ABC','DEEEEF','FHGIJ'])
x, *q, y = list(g) # TODO: auto-unroll as in Python
print x, y, q #: ('B', 'A', 'C') ('HGI', 'F', 'J') [('EEEE', 'D', 'F')]

#%% assign_shadow,barebones
a = 5
print a #: 5
a : str = 's'
print a #: s

#%% assign_err_must_exist,barebones
a = 1
def foo():
    a += 2 #! local variable 'a' referenced before assignment

#%% assign_rename,barebones
y = int
z = y(5)
print z #: 5

def foo(x): return x + 1
x = foo
print x(1) #: 2

#%% assign_err_6,barebones
x = bar #! name 'bar' is not defined

#%% assign_err_7,barebones
foo() += bar #! cannot assign to given expression

#%% assign_update_eq,barebones
a = 5
a += 3
print a #: 8
a -= 1
print a #: 7

class Foo:
    a: int
    def __add__(self, i: int):
        print 'add!'
        return Foo(self.a + i)
    def __iadd__(self, i: int):
        print 'iadd!'
        self.a += i
        return self
    def __str__(self):
        return str(self.a)
f = Foo(3)
print f + 2 #: add!
#: 5
f += 6 #: iadd!
print f #: 9

#%% del,barebones
a = 5
del a
print a #! name 'a' is not defined

#%% del_index,barebones
y = [1, 2]
del y[0]
print y #: [2]

#%% del_error,barebones
a = [1]
del a.ptr #! cannot delete given expression

#%% assert,barebones
assert True
assert True, "blah"

try:
    assert False
except AssertionError as e:
    print e.message[:15], e.message[-21:] #: Assert failed ( simplify_stmt.hs:174)

try:
    assert False, f"hehe {1}"
except AssertionError as e:
    print e.message[:23], e.message[-21:] #: Assert failed: hehe 1 ( simplify_stmt.hs:179)

#%% print,barebones
print 1,
print 1, 2  #: 1 1 2

print 1, 2  #: 1 2
print(3, "4", sep="-", end=" !\n") #: 3-4 !

print(1, 2) #: 1 2
print (1, 2) #: (1, 2)

def foo(i, j):
    return i + j
print 3 |> foo(1)  #: 4

#%% return_fail,barebones
return #! 'return' outside function

#%% yield_fail,barebones
yield 5 #! 'yield' outside function

#%% yield_fail_2,barebones
(yield) #! 'yield' outside function

#%% while_else,barebones
a = 1
while a:
    print a #: 1
    a -= 1
else:
    print 'else' #: else
a = 1
while a:
    print a #: 1
    a -= 1
else not break:
    print 'else' #: else
while True:
    print 'infinite' #: infinite
    break
else:
    print 'nope'

#%% for_assignment,barebones
l = [[1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]
for a, *m, b in l:
    print a + b, len(m)
#: 5 2
#: 14 3
#: 21 0

#%% for_else,barebones
for i in [1]:
    print i #: 1
else:
    print 'else' #: else
for i in [1]:
    print i #: 1
else not break:
    print 'else' #: else
for i in [1]:
    print i #: 1
    break
else:
    print 'nope'

best = 4
for s in [3, 4, 5]:
    for i in [s]:
        if s >= best:
            print('b:', best)
            break
    else:
        print('s:', s)
        best = s
#: s: 3
#: b: 3
#: b: 3

#%% match
def foo(x):
    match x:
        case 1:
            print 'int'
        case 2 ... 10:
            print 'range'
        case 'ACGT':
            print 'string'
        case (a, 1):
            print 'tuple_wild', a
        case []:
            print 'list'
        case [[]]:
            print 'list list'
        case [1, 2]:
            print 'list 2'
        case [1, z, ...] if z < 5:
            print 'list 3', z
        case [1, _, ..., zz] | (1, zz):
            print 'list 4', zz
        case (1 ... 10, s := ('ACGT', 1 ... 4)):
            print 'complex', s
        case _:
            print 'else'
foo(1) #: int
foo(5) #: range
foo('ACGT') #: string
foo((9, 1)) #: tuple_wild 9
foo(List[int]()) #: list
foo([List[int]()]) #: list list
foo([1, 2]) #: list 2
foo([1, 3]) #: list 3 3
foo([1, 5]) #: else
foo([1, 5, 10]) #: list 4 10
foo((1, 33)) #: list 4 33
foo((9, ('ACGT', 3))) #: complex ('ACGT', 3)
foo(range(10)) #: else

for op in 'MI=DXSN':
    match op:
        case 'M' | '=' | 'X':
            print('case 1')
        case 'I' or 'S':
            print('case 2')
        case _:
            print('case 3')
#: case 1
#: case 2
#: case 1
#: case 3
#: case 1
#: case 2
#: case 3

#%% match_err_1,barebones
match [1, 2]:
    case [1, ..., 2, ..., 3]: pass
#! multiple ellipses in a pattern

#%% global,barebones
a = 1
def foo():
    global a
    a += 1
print a,
foo()
print a  #: 1 2

#%% global_err,barebones
a = 1
global a #! 'global' outside function

#%% global_err_2,barebones
def foo():
    global b #! name 'b' is not defined

#%% global_err_3,barebones
def foo():
    b = 1
    def bar():
        global b #! no binding for global 'b' found

#%% global_err_4,barebones
a = 1
def foo():
    a += 1
foo()  #! local variable 'a' referenced before assignment

#%% global_ref,barebones
a = [1]
def foo():
    a.append(2)
foo()
print a #: [1, 2]

#%% yield_from,barebones
def foo():
    yield from range(3)
    yield from range(10, 13)
    yield -1
print list(foo())  #: [0, 1, 2, 10, 11, 12, -1]

#%% with,barebones
class Foo:
    i: int
    def __enter__(self: Foo):
        print '> foo! ' + str(self.i)
    def __exit__(self: Foo):
        print '< foo! ' + str(self.i)
    def foo(self: Foo):
        print 'woof'
class Bar:
    s: str
    def __enter__(self: Bar):
        print '> bar! ' + self.s
    def __exit__(self: Bar):
        print '< bar! ' + self.s
    def bar(self: Bar):
        print 'meow'
with Foo(0) as f:
#: > foo! 0
    f.foo()  #: woof
#: < foo! 0
with Foo(1) as f, Bar('s') as b:
#: > foo! 1
#: > bar! s
    f.foo()  #: woof
    b.bar()  #: meow
#: < bar! s
#: < foo! 1
with Foo(2), Bar('t') as q:
#: > foo! 2
#: > bar! t
    print 'eeh'  #: eeh
    q.bar()  #: meow
#: < bar! t
#: < foo! 2

#%% import_c,barebones
from C import sqrt(float) -> float
print sqrt(4.0) #: 2

from C import puts(cobj)
puts("hello".ptr) #: hello

from C import atoi(cobj) -> int as s2i
print s2i("11".ptr) #: 11

@C
def log(x: float) -> float:
    pass
print log(5.5)  #: 1.70475

from C import seq_flags: Int[32] as e
# debug | standalone == 5
print e  #: 5

#%% import_c_shadow_error,barebones
# Issue #45
from C import sqrt(float) -> float as foo
sqrt(100.0)  #! name 'sqrt' is not defined


#%% import_c_dylib,barebones
from internal.dlopen import dlext
RT = "./libherculesrt." + dlext()
if RT[-3:] == ".so":
    RT = "build/" + RT[2:]
from C import RT.seq_str_int(int, str, Ptr[bool]) -> str as sp
p = False
print sp(65, "", __ptr__(p))  #: 65

#%% import_c_dylib_error,barebones
from C import "".seq_print(str) as sp
sp("hi!") #! syntax error, unexpected '"'

#%% import,barebones
zoo, _zoo = 1, 1
print zoo, _zoo, __name__  #: 1 1 __main__

import a  #: a
a.foo() #: a.foo

from a import foo, bar as b
foo() #: a.foo
b() #: a.bar

print str(a)[:9], str(a)[-15:] #: <module ' a/__init__.hs'>

import a.b
print a.b.c #: a.b.c
a.b.har() #: a.b.har a.b.__init__ a.b.c

print a.b.A.B.b_foo().__add__(1) #: a.b.A.B.b_foo()
#: 2

print str(a.b)[:9], str(a.b)[-17:] #: <module ' a/b/__init__.hs'>
print Int[a.b.stt].__class__.__name__  #: Int[5]

from a.b import *
har() #: a.b.har a.b.__init__ a.b.c
a.b.har() #: a.b.har a.b.__init__ a.b.c
fx() #: a.foo
print(stt, Int[stt].__class__.__name__)  #: 5 Int[5]

from a import *
print zoo, _zoo, __name__  #: 5 1 __main__

f = Foo(Ptr[B]())
print f.__class__.__name__, f.t.__class__.__name__  #: Foo Ptr[B]

a.ha()  #: B

print par  #: x

#%% import_order,barebones
def foo():
    import a
    a.foo()
def bar():
    import a
    a.bar()

bar() #: a
#: a.bar
foo() #: a.foo

#%% import_class
import sys
print str(sys)[:20]  #: <module 'sys' from '
print sys.maxsize  #: 9223372036854775807

#%% import_rec,barebones
from a.b.rec1 import bar
#: import rec1
#: import rec2
#: done rec2
#: rec2.x
#: done rec1
bar()
#: rec1.bar

#%% import_rec_err,barebones
from a.b.rec1_err import bar
#! cannot import name 'bar' from 'a.b.rec1_err'
#! name 'bar' is not defined
# TODO: get rid of this!
#! no module named 'rec2_err'

#%% import_err_1,barebones
class Foo:
    import bar #! unexpected expression in class definition

#%% import_err_2,barebones
import "".a.b.c #! syntax error, unexpected '"'

#%% import_err_3,barebones
from a.b import foo() #! function signatures only allowed when importing C or Python functions

#%% import_err_4,barebones
from a.b.c import hai.hey #! expected identifier

#%% import_err_4_x,barebones
import whatever #! no module named 'whatever'

#%% import_err_5,barebones
import a.b
print a.b.x #! cannot import name 'x' from 'a.b.__init__'

#%% import_err_6,barebones
from a.b import whatever #! cannot import name 'whatever' from 'a.b.__init__'

#%% function_err_0,barebones
def foo(a, b, a):
    pass #! duplicate argument 'a' in function definition

#%% function_err_0b,barebones
def foo(a, b=1, c):
    pass #! non-default argument 'c' follows default argument

#%% function_err_0b_ok,barebones
def foo(a, b=1, *c):
    pass

#%% function_err_0c,barebones
def foo(a, b=1, *c, *d):
    pass #! multiple star arguments provided

#%% function_err_0e,barebones
def foo(a, b=1, *c = 1):
    pass #! star arguments cannot have default values

#%% function_err_0f,barebones
def foo(a, b=1, **c, **kwargs):
    pass #! kwargs must be the last argument

#%% function_err_0h,barebones
def foo(a, b=1, **c = 1):
    pass #! star arguments cannot have default values

#%% function_err_0i,barebones
def foo(a, **c, d):
    pass #! kwargs must be the last argument

#%% function_err_1,barebones
def foo():
    @__force__
    def bar(): pass #! builtin function must be a top-level statement

#%% function_err_2,barebones
def f[T: Static[float]]():
    pass
#! expected 'int' or 'str' (only integers and strings can be static)

#%% function_err_3,barebones
def f(a, b=a):
    pass
#! name 'a' is not defined

#%% function_llvm_err_1,barebones
@llvm
def foo():
    blah
#! return types required for LLVM and C functions

#%% function_llvm_err_2,barebones
@llvm
def foo() -> int:
    a{={=}}
#! invalid LLVM code

#%% function_llvm_err_4,barebones
a = 5
@llvm
def foo() -> int:
    a{=a
#! invalid LLVM code

#%% function_self,barebones
class Foo:
    def foo(self):
        return 'F'
f = Foo()
print f.foo() #: F

#%% function_self_err,barebones
class Foo:
    def foo(self):
        return 'F'
Foo.foo(1) #! 'Foo' object has no method 'foo' with arguments (int)

#%% function_nested,barebones
def foo(v):
    value = v
    def bar():
        return value
    return bar
baz = foo(2)
print baz() #: 2

def f(x):
    a=1
    def g(y):
        return a+y
    return g(x)
print f(5) #: 6

#%% nested_generic_static,barebones
def foo():
    N: Static[int] = 5
    Z: Static[int] = 15
    T = Int[Z]
    def bar():
        x = __array__[T](N)
        print(x.__class__.__name__)
    return bar
foo()()  #: Array[Int[15]]

def f[T]():
    def g():
        return T()
    return g()
print f(int) #: 0

#%% class_err_1,barebones
@extend
@foo
class Foo:
    pass
#! cannot combine '@extend' with other attributes or decorators

#%% class_err_1b,barebones
size_t = i32
@extend
class size_t:
    pass
#! class name 'size_t' is not defined

#%% class_err_2,barebones
def foo():
    @extend
    class Foo:
        pass
#! class extension must be a top-level statement

#%% class_nested,barebones
class Foo:
    foo: int
    class Bar:
        bar: int
        b: Optional[Foo.Bar]
        c: Optional[int]
        class Moo:
            # TODO: allow nested class reference to the upclass
            # x: Foo.Bar
            x: int
y = Foo(1)
z = Foo.Bar(2, None, 4)
m = Foo.Bar.Moo(5)
print y.foo #: 1
print z.bar, z.b.__bool__(), z.c, m.x  #: 2 False 4 5

#%% class_nested_2,barebones
@tuple
class Foo:
    @tuple
    class Bar:
        x: int
    x: int
    b: Bar
    c: Foo.Bar
f = Foo(5, Foo.Bar(6), Foo.Bar(7))
print(f) #: (x: 5, b: (x: 6), c: (x: 7))

#%% class_nested_err,barebones
class Foo:
    class Bar:
        b: Ptr[Bar]
#! name 'Bar' is not defined

#%% class_err_4,barebones
@extend
class Foo:
    pass
#! class name 'Foo' is not defined

#%% class_err_5,barebones
class Foo[T, U]:
    pass
@extend
class Foo[T]:
    pass
#! class extensions cannot define data attributes and generics or inherit other classes

#%% class_err_7,barebones
class Foo:
    a: int
    a: int
#! duplicate data attribute 'a' in class definition

#%% class_err_tuple_no_recursive,barebones
@tuple
class Foo:
    a: Foo
#! name 'Foo' is not defined

#%% class_err_8,barebones
class Foo:
    while True: pass
#! unexpected expression in class definition

#%% class_err_9,barebones
class F[T: Static[float]]:
    pass
#! expected 'int' or 'str' (only integers and strings can be static)

#%% class_err_10,barebones
def foo[T]():
    class A:
        x: T
#! name 'T' cannot be captured

#%% class_err_11,barebones
def foo(x):
    class A:
        def bar():
            print x
#! name 'x' cannot be captured

#%% class_err_12,barebones
def foo(x):
    T = type(x)
    class A:
        def bar():
            print T()
#! name 'T' cannot be captured

#%% recursive_class,barebones
class Node[T]:
    data: T
    children: List[Node[T]]
    def __init__(self, data: T):
        self.data = data
        self.children = []
print Node(2).data #: 2

class Node2:
    data: int
    children: List[Node2]
    def __init__(self, data: int):
        self.data = data
        self.children = []
print Node2(3).data #: 3

#%% class_auto_init,barebones
class X[T]:
    a: int = 4
    b: int
    c: T
    d: str = 'oops'
    def __str__(self):
        return f'X({self.a},{self.b},{self.c},{self.d})'
x = X[float]()
print x #: X(4,0,0,oops)
y = X(c='darius',a=5)
print y #: X(5,0,darius,oops)

#%% magic,barebones
@tuple
class Foo:
    x: int
    y: int
a, b = Foo(1, 2), Foo(1, 3)
print a, b #: (x: 1, y: 2) (x: 1, y: 3)
print a.__len__() #: 2
print a.__hash__(), b.__hash__() #: 175247769363 175247769360
print a == a, a == b #: True False
print a != a, a != b #: False True
print a < a, a < b, b < a #: False True False
print a <= a, a <= b, b <= a #: True True False
print a > a, a > b, b > a #: False False True
print a >= a, a >= b, b >= a #: True False True
print a.__getitem__(1)  #: 2
print list(a.__iter__()) #: [1, 2]

#%% magic_class,barebones
@dataclass(eq=True, order=True)
class Foo:
    x: int
    y: int
    def __str__(self): return f'{self.x}_{self.y}'
a, b = Foo(1, 2), Foo(1, 3)
print a, b #: 1_2 1_3
print a == a, a == b #: True False
print a != a, a != b #: False True
print a < a, a < b, b < a #: False True False
print a <= a, a <= b, b <= a #: True True False
print a > a, a > b, b > a #: False False True
print a >= a, a >= b, b >= a #: True False True

# Right magic test
class X:
    x: int
class Y:
    y: int
    def __eq__(self, o: X): return self.y == o.x
    def __ne__(self, o: X): return self.y != o.x
    def __le__(self, o: X): return self.y <= o.x
    def __lt__(self, o: X): return self.y <  o.x
    def __ge__(self, o: X): return self.y >= o.x
    def __gt__(self, o: X): return self.y >  o.x
    def __add__(self, o: X):  return self.y + o.x + 1
    def __radd__(self, o: X): return self.y + o.x + 2
print Y(1) == X(1), Y(1) != X(1)  #: True False
print X(1) == Y(1), X(1) != Y(1)  #: True False
print Y(1) <= X(2), Y(1) < X(2)  #: True True
print X(1) <= Y(2), X(1) < Y(2)  #: True True
print Y(1) >= X(2), Y(1) > X(2)  #: False False
print X(1) >= Y(2), X(1) > Y(2)  #: False False
print X(1) + Y(2)  #: 5
print Y(1) + X(2)  #: 4


class A:
    def __radd__(self, n: int):
        return 0
def f():
    print('f')
    return 1
def g():
    print('g')
    return A()
f() + g()
#: f
#: g

#%% magic_2,barebones
@tuple
class Foo:
    pass
a, b = Foo(), Foo()
print a, b #: () ()
print a.__len__() #: 0
print a.__hash__(), b.__hash__() #: 0 0
print a == a, a == b #: True True
print a != a, a != b #: False False
print a < a, a < b, b < a #: False False False
print a <= a, a <= b, b <= a #: True True True
print a > a, a > b, b > a #: False False False
print a >= a, a >= b, b >= a #: True True True

# TODO: pickle / to_py / from_py

#%% magic_contains,barebones
sponge = (1, 'z', 1.55, 'q', 48556)
print 1.1 in sponge #: False
print 'q' in sponge #: True
print True in sponge #: False

bob = (1, 2, 3)
print 1.1 in sponge #: False
print 1 in sponge #: True
print 0 in sponge #: False

#%% magic_err_2,barebones
@tuple
class Foo:
    pass
try:
    print Foo().__getitem__(1)
except IndexError:
    print 'error'  #: error

#%% magic_empty_tuple,barebones
@tuple
class Foo:
    pass
print list(Foo().__iter__())  #: []

#%% magic_err_4,barebones
@tuple(eq=False)
class Foo:
    x: int
Foo(1).__eq__(Foo(1)) #! 'Foo' object has no attribute '__eq__'

#%% magic_err_5,barebones
@tuple(pickle=False)
class Foo:
    x: int
p = Ptr[byte]()
Foo(1).__pickle__(p) #! 'Foo' object has no attribute '__pickle__'

#%% magic_err_6,barebones
@tuple(container=False)
class Foo:
    x: int
Foo(1).__getitem__(0) #! 'Foo' object has no attribute '__getitem__'

#%% magic_err_7,barebones
@tuple(python=False)
class Foo:
    x: int
p = Ptr[byte]()
Foo(1).__to_py__(p) #! 'Foo' object has no attribute '__to_py__'

#%% python
from python import os
print os.name  #: posix

from python import datetime
z = datetime.datetime.utcfromtimestamp(0)
print z  #: 1970-01-01 00:00:00

#%% python_numpy
from python import numpy as np
a = np.arange(9).reshape(3, 3)
print a
#: [[0 1 2]
#:  [3 4 5]
#:  [6 7 8]]
print a.dtype.name  #: int64
print np.transpose(a)
#: [[0 3 6]
#:  [1 4 7]
#:  [2 5 8]]
n = np.array([[1, 2], [3, 4]])
print n[0], n[0][0] + 1 #: [1 2] 2

a = np.array([1,2,3])
print(a + 1) #: [2 3 4]
print(a - 1) #: [0 1 2]
print(1 - a) #: [ 0 -1 -2]

#%% python_import_fn
from python import re.split(str, str) -> List[str] as rs
print rs(r'\W+', 'Words, words, words.')  #: ['Words', 'words', 'words', '']

#%% python_import_fn_2
from python import os.system(str) -> int
system("echo 'hello!'")  #: hello!

#%% python_pydef
@python
def test_pydef(n) -> str:
    return ''.join(map(str,range(n)))
print test_pydef(5)  #: 01234

#%% python_pydef_nested
def foo():
    @python
    def pyfoo():
        return 1
    print pyfoo() #: 1
    if True:
        @python
        def pyfoo2():
            return 2
        print pyfoo2() #: 2
    pass
    @python
    def pyfoo3():
        if 1:
            return 3
    return str(pyfoo3())
print foo() #: 3

#%% python_pyobj
@python
def foofn() -> Dict[pyobj, pyobj]:
    return {"str": "hai", "int": 1}

foo = foofn()
print(sorted(foo.items(), key=lambda x: str(x)), foo.__class__.__name__)
#: [('int', 1), ('str', 'hai')] Dict[pyobj,pyobj]
foo["hs"] = 5.15
print(sorted(foo.items(), key=lambda x: str(x)), foo["hs"].__class__.__name__, foo.__class__.__name__)
#: [('hs', 5.15), ('int', 1), ('str', 'hai')] pyobj Dict[pyobj,pyobj]

a = {1: "s", 2: "t"}
a[3] = foo["str"]
print(sorted(a.items()))  #: [(1, 's'), (2, 't'), (3, 'hai')]


#%% python_isinstance
import python

@python
def foo():
    return 1

z = foo()
print(z.__class__.__name__)  #: pyobj

print isinstance(z, pyobj)  #: True
print isinstance(z, int)  #: False
print isinstance(z, python.int)  #: True
print isinstance(z, python.ValueError)  #: False

print isinstance(z, (int, str, python.int))  #: True
print isinstance(z, (int, str, python.AttributeError))  #: False

try:
    foo().x
except python.ValueError:
    pass
except python.AttributeError as e:
    print('caught', e, e.__class__.__name__) #: caught 'int' object has no attribute 'x' pyobj


#%% python_exceptions
import python

@python
def foo():
    return 1

try:
    foo().x
except python.AttributeError as f:
    print 'py.Att', f  #: py.Att 'int' object has no attribute 'x'
except ValueError:
    print 'Val'
except PyError as e:
    print 'PyError', e
try:
    foo().x
except python.ValueError as f:
    print 'py.Att', f
except ValueError:
    print 'Val'
except PyError as e:
    print 'PyError', e  #: PyError 'int' object has no attribute 'x'
try:
    raise ValueError("ho")
except python.ValueError as f:
    print 'py.Att', f
except ValueError:
    print 'Val'  #: Val
except PyError as e:
    print 'PyError', e


#%% typeof_definition_error,barebones
a = 1
class X:
    b: type(a) #! cannot use type() in type signatures

#%% typeof_definition_error_2,barebones
def foo(a)->type(a): pass #! cannot use type() in type signatures

#%% typeof_definition_error_3,barebones
a=1
b: type(a) = 1 #! cannot use type() in type signatures

#%% assign_underscore,barebones
_ = 5
_ = 's'

#%% inherit_class_4,barebones
class defdict[K,V](Static[Dict[K,V]]):
    fx: Function[[],V]
    def __init__(self, d: Dict[K,V], fx: Function[[], V]):
        self.__init__()
        for k,v in d.items(): self[k] = v
        self.fx = fx
    def __getitem__(self, key: K) -> V:
        if key in self:
            return self.values[self.keys.index(key)]
        else:
            self[key] = self.fx()
            return self[key]
z = defdict({'ha':1}, lambda: -1)
print z
print z['he']
print z
#: {'ha': 1}
#: -1
#: {'ha': 1, 'he': -1}

class Foo:
    x: int
    def foo(self):
        return f'foo {self.x}'
class Bar[T]:
    y: T
    def bar(self):
        return f'bar {self.y}/{self.y.__class__.__name__}'
class FooBarBaz[T](Static[Foo], Static[Bar[T]]):
    def baz(self):
        return f'baz! {self.foo()} {self.bar()}'
print FooBarBaz[str]().foo() #: foo 0
print FooBarBaz[float]().bar() #: bar 0/float
print FooBarBaz[str]().baz() #: baz! foo 0 bar /str

#%% inherit_class_err_5,barebones
class defdict(Static[Dict[str,float]]):
    def __init__(self, d: Dict[str, float]):
        self.__init__(d.items())
z = defdict()
z[1.1] #! 'defdict' object has no method '__getitem__' with arguments (defdict, float)

#%% inherit_tuple,barebones
class Foo:
    a: int
    b: str
    def __init__(self, a: int):
        self.a, self.b = a, 'yoo'
@tuple
class FooTup(Static[Foo]): pass

f = Foo(5)
print f.a, f.b #: 5 yoo
fp = FooTup(6, 's')
print fp #: (a: 6, b: 's')

#%% inherit_class_err_1,barebones
class defdict(Static[Array[int]]):
    pass #! reference classes cannot inherit tuple classes

#%% inherit_class_err_2,barebones
@tuple
class defdict(Static[int]):
    pass #! internal classes cannot inherit other classes

#%% inherit_class_err_3,barebones
class defdict(Static[Dict[int, float, float]]):
    pass #! Dict takes 2 generics (3 given)

#%% inherit_class_err_4,barebones
class Foo:
    x: int
class Bar:
    x: float
class FooBar(Static[Foo], Static[Bar]):
    pass
# right now works as we rename other fields

#%% keyword_prefix,barebones
def foo(return_, pass_, yield_, break_, continue_, print_, assert_):
    return_.append(1)
    pass_.append(2)
    yield_.append(3)
    break_.append(4)
    continue_.append(5)
    print_.append(6)
    assert_.append(7)
    return return_, pass_, yield_, break_, continue_, print_, assert_
print foo([1], [1], [1], [1], [1], [1], [1])
#: ([1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7])


#%% class_deduce,barebones
@deduce
class Foo:
    def __init__(self, x):
        self.x = [x]
        self.y = 1, x

f = Foo(1)
print(f.x, f.y, f.__class__.__name__) #: [1] (1, 1) Foo[List[int],Tuple[int,int]]

f: Foo = Foo('s')
print(f.x, f.y, f.__class__.__name__) #: ['s'] (1, 's') Foo[List[str],Tuple[int,str]]

@deduce
class Bar:
    def __init__(self, y):
        self.y = Foo(y)

b = Bar(3.1)
print(b.y.x, b.__class__.__name__) #: [3.1] Bar[Foo[List[float],Tuple[int,float]]]

#%% multi_error,barebones
a = 55
print z  #! name 'z' is not defined
print(a, q, w)  #! name 'q' is not defined
print quit  #! name 'quit' is not defined

#%% class_var,barebones
class Foo:
    cx = 15
    x: int = 10
    cy: ClassVar[str] = "ho"
    class Bar:
        bx = 1.1
print(Foo.cx)  #: 15
f = Foo()
print(Foo.cy, f.cy)  #: ho ho
print(Foo.Bar.bx)  #: 1.1

Foo.cx = 10
print(Foo.cx)  #: 10

def x():
    class Foo:
        i = 0
        f = Foo()
        def __init__(self):
            Foo.i += 1
        def __repr__(self):
            return 'heh-cls'
    Foo(), Foo(), Foo()
    print Foo.f, Foo.i  #: heh-cls 4
    return Foo()
f = x()
print f.f, f.i  #: heh-cls 5

@tuple
class Fot:
    f = Fot()
    def __repr__(self):
        return 'heh-tup'
print Fot.f  #: heh-tup


#%% dot_access_error,barebones
class Foo:
    x: int = 1
Foo.x #! 'Foo' object has no attribute 'x'

#%% scoping_same_name,barebones
def match(pattern: str, string: str, flags: int = 0):
    pass

def match(match):
    if True:
        match = 0
    match

match(1)

#%% loop_domination,barebones
for i in range(2):
    try: dat = 1
    except: pass
    print(dat)
#: 1
#: 1

def comprehension_test(x):
    for n in range(3):
        print('>', n)
    l = ['1', '2', str(x)]
    x = [n for n in l]
    print(x, n)
comprehension_test(5)
#: > 0
#: > 1
#: > 2
#: ['1', '2', '5'] 2


#%% block_unroll,barebones
# Ensure that block unrolling is done in RAII manner on error
def foo():
    while True:
        def magic(a: x):
            return
        print b
foo()
#! name 'x' is not defined
#! name 'b' is not defined

#%% capture_recursive,barebones
def f(x: int) -> int:
    z = 2 * x
    def g(y: int) -> int:
        if y == 0:
            return 1
        else:
            return g(y - 1) * z
    return g(4)
print(f(3))  #: 1296

#%% class_setter,barebones
class Foo:
    _x: int

    @property
    def x(self):
        print('getter')
        return self._x

    @x.setter
    def x(self, v):
        print('setter')
        self._x = v

f = Foo(1)
print(f.x)
#: getter
#: 1

f.x = 99
print(f.x)
print(f._x)
#: setter
#: getter
#: 99
#: 99
