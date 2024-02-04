#%% expr,barebones
a = 5; b = 3
print a, b  #: 5 3

#%% assign_optional,barebones
a = None
print a  #: None
a = 5
print a  #: 5

b: Optional[float] = Optional[float](6.5)
c: Optional[float] = 5.5
print b, c #: 6.5 5.5

#%% assign_type_alias,barebones
I = int
print I(5) #: 5

L = Dict[int, str]
l = L()
print l #: {}
l[5] = 'haha'
print l #: {5: 'haha'}

#%% assign_type_annotation,barebones
a: List[int] = []
print a  #: []

#%% assign_type_err,barebones
a = 5
if 1:
    a = 3.3  #! 'float' does not match expected type 'int'
a

#%% assign_atomic,barebones
i = 1
f = 1.1

@llvm
def xchg(d: Ptr[int], b: int) -> None:
    %tmp = atomicrmw xchg i64* %d, i64 %b seq_cst
    ret {} {}
@llvm
def aadd(d: Ptr[int], b: int) -> int:
    %tmp = atomicrmw add i64* %d, i64 %b seq_cst
    ret i64 %tmp
@llvm
def amin(d: Ptr[int], b: int) -> int:
    %tmp = atomicrmw min i64* %d, i64 %b seq_cst
    ret i64 %tmp
@llvm
def amax(d: Ptr[int], b: int) -> int:
    %tmp = atomicrmw max i64* %d, i64 %b seq_cst
    ret i64 %tmp
def min(a, b): return a if a < b else b
def max(a, b): return a if a > b else b

@extend
class int:
    def __atomic_xchg__(self: Ptr[int], i: int):
        print 'atomic:', self[0], '<-', i
        xchg(self, i)
    def __atomic_add__(self: Ptr[int], i: int):
        print 'atomic:', self[0], '+=', i
        return aadd(self, i)
    def __atomic_min__(self: Ptr[int], b: int):
        print 'atomic:', self[0], '<?=', b
        return amin(self, b)
    def __atomic_max__(self: Ptr[int], b: int):
        print 'atomic:', self[0], '>?=', b
        return amax(self, b)

@atomic
def foo(x):
    global i, f

    i += 1 #: atomic: 1 += 1
    print i #: 2
    i //= 2 #: atomic: 2 <- 1
    print i #: 1
    i = 3 #: atomic: 1 <- 3
    print i #: 3
    i = min(i, 10) #: atomic: 3 <?= 10
    print i #: 3
    i = max(i, 10) #: atomic: 3 >?= 10
    print i #: 10
    i = max(20, i) #: atomic: 10 <- 20
    print i #: 20

    f += 1.1
    f = 3.3
    f = max(f, 5.5)
foo(1)
print i, f #: 20 5.5

#%% assign_atomic_real
i = 1
f = 1.1
@atomic
def foo(x):
    global i, f

    i += 1
    print i #: 2
    i //= 2
    print i #: 1
    i = 3
    print i #: 3
    i = min(i, 10)
    print i #: 3
    i = max(i, 10)
    print i #: 10

    f += 1.1
    f = 3.3
    f = max(f, 5.5)
foo(1)
print i, f #: 10 5.5

#%% assign_member,barebones
class Foo:
    x: Optional[int]
f = Foo()
print f.x #: None
f.x = 5
print f.x #: 5

fo = Optional(Foo())
fo.x = 6
print fo.x #: 6

#%% assign_member_err_1,barebones
class Foo:
    x: Optional[int]
Foo().y = 5 #! 'Foo' object has no attribute 'y'

#%% assign_member_err_2,barebones
@tuple
class Foo:
    x: Optional[int]
Foo().x = 5 #! cannot modify tuple attributes

#%% return,barebones
def foo():
    return 1
print foo()  #: 1

def bar():
    print 2
    return
    print 1
bar()  #: 2

#%% yield,barebones
def foo():
    yield 1
print [i for i in foo()], str(foo())[:16]  #: [1] <generator at 0x

#%% yield_void,barebones
def foo():
    yield
    print 1
y = foo()
print y.done()  #: False
y.next()  #: 1
# TODO: next() should work here!
print y.done()  #: True

#%% yield_return,barebones
def foo():
    yield 1
    return
    yield 2
print list(foo())  #: [1]

def foo(x=0):
    yield 1
    if x:
        return
    yield 2
print list(foo())  #: [1, 2]
print list(foo(1))  #: [1]

def foo(x=0):
    if x:
        return
    yield 1
    yield 2
print list(foo())  #: [1, 2]
print list(foo(1))  #: []

#%% return_none_err_1,barebones
def foo(n: int):
    if n > 0:
        return
    else:
        return 1
foo(1)
#! 'NoneType' does not match expected type 'int'
#! during the realization of foo(n: int)

#%% return_none_err_2,barebones
def foo(n: int):
    if n > 0:
        return 1
    return
foo(1)
#! 'int' does not match expected type 'NoneType'
#! during the realization of foo(n: int)

#%% while,barebones
a = 3
while a:
    print a
    a -= 1
#: 3
#: 2
#: 1

#%% for_break_continue,barebones
for i in range(10):
    if i % 2 == 0:
        continue
    print i
    if i >= 5:
        break
#: 1
#: 3
#: 5

#%% for_error,barebones
for i in 1:
    pass
#! 'int' object has no attribute '__iter__'

#%% for_void,barebones
def foo(): yield
for i in foo():
    print i.__class__.__name__  #: NoneType

#%% if,barebones
for a, b in [(1, 2), (3, 3), (5, 4)]:
    if a > b:
        print '1',
    elif a == b:
        print '=',
    else:
        print '2',
print '_'  #: 2 = 1 _

if 1:
    print '1' #: 1

#%% static_if,barebones
def foo(x, N: Static[int]):
    if isinstance(x, int):
        return x + 1
    elif isinstance(x, float):
        return x.__pow__(.5)
    elif isinstance(x, Tuple[int, str]):
        return f'foo: {x[1]}'
    elif isinstance(x, Tuple) and (N >= 3 or staticlen(x) > 2):
        return x[2:]
    elif hasattr(x, '__len__'):
        return 'len ' + str(x.__len__())
    else:
        compile_error('invalid type')
print foo(N=1, x=1) #: 2
print foo(N=1, x=2.0) #: 1.41421
print foo(N=1, x=(1, 'bar')) #: foo: bar
print foo(N=1, x=(1, 2)) #: len 2
print foo(N=3, x=(1, 2)) #: ()
print foo(N=1, x=(1, 2, 3)) #: (3,)

#%% try_throw,barebones
class MyError(Static[Exception]):
    def __init__(self, message: str):
        super().__init__('MyError', message)
try:
    raise MyError("hello!")
except MyError as e:
    print str(e)  #: hello!
try:
    raise OSError("hello os!")
# TODO: except (MyError, OSError) as e:
#     print str(e)
except MyError:
    print "my"
except OSError as o:
    print "os", o.typename, len(o.message), o.file[-17:], o.line
    #: os OSError 9 typecheck_stmt.hs 284
finally:
    print "whoa"  #: whoa

# Test function name
def foo():
    raise MyError("foo!")
try:
    foo()
except MyError as e:
    print e.typename, e.message #: MyError foo!

#%% throw_error,barebones
raise 'hello'
#! exceptions must derive from BaseException

#%% function_builtin_error,barebones
@__force__
def foo(x):
    pass
#! builtin, exported and external functions cannot be generic

#%% extend,barebones
@extend
class int:
    def run_lola_run(self):
        while self > 0:
            yield self
            self -= 1
print list((5).run_lola_run())  #: [5, 4, 3, 2, 1]


#%% early_return,barebones
def foo(x):
    print  x-1
    return
    print len(x)
foo(5) #: 4

def foo2(x):
    if isinstance(x, int):
        print  x+1
        return
    print len(x)
foo2(1) #: 2
foo2('s') #: 1

#%% superf,barebones
class Foo:
    def foo(a):
        # superf(a)
        print 'foo-1', a
    def foo(a: int):
        superf(a)
        print 'foo-2', a
    def foo(a: str):
        superf(a)
        print 'foo-3', a
    def foo(a):
        superf(a)
        print 'foo-4', a
Foo.foo(1)
#: foo-1 1
#: foo-2 1
#: foo-4 1

class Bear:
    def woof(x):
        return f'bear woof {x}'
@extend
class Bear:
    def woof(x):
        return superf(x) + f' bear w--f {x}'
print Bear.woof('!')
#: bear woof ! bear w--f !

class PolarBear(Static[Bear]):
    def woof():
        return 'polar ' + superf('@')
print PolarBear.woof()
#: polar bear woof @ bear w--f @

#%% superf_error,barebones
class Foo:
    def foo(a):
        superf(a)
        print 'foo-1', a
Foo.foo(1)
#! no superf methods found
#! during the realization of foo(a: int)

#%% staticmethod,barebones
class Foo:
    def __repr__(self):
        return 'Foo'
    def m(self):
        print 'm', self
    @staticmethod
    def sm(i):
        print 'sm', i
Foo.sm(1)  #: sm 1
Foo().sm(2)  #: sm 2
Foo().m()  #: m Foo
