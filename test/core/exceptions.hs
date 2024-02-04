class Exc1(Static[Exception]):
    def __init__(self, msg: str):
        super().__init__('Exc1', msg)

    def show(self):
        print self.message

class Exc2(Static[Exception]):
    def __init__(self, msg: str):
        super().__init__('Exc2', msg)

    def show(self):
        print self.message

class A(Static[Exception]):
    def __init__(self, msg: str):
        super().__init__('A', msg)

class B(Static[Exception]):
    def __init__(self, msg: str):
        super().__init__('B', msg)

class C(Static[Exception]):
    def __init__(self, msg: str):
        super().__init__('C', msg)

class D(Static[Exception]):
    def __init__(self, msg: str):
        super().__init__('D', msg)

class E(Static[Exception]):
    def __init__(self, msg: str):
        super().__init__('E', msg)

def foo1(x):
    if x:
        raise Exc1('e1')
    else:
        raise Exc2('e2')

def foo2(x):
    foo1(x)

def foo(x):
    foo2(x)

def square(x):
    return x * x

def bar(x):
    try:
        print 'try'
        foo(x)
        print 'error'
    except Exc1 as e:
        print 'catch Exc1 1'
        print e.message
        print 'catch Exc1 2'
    except Exc2 as e:
        print 'catch Exc2 1'
        print e.message
        print 'catch Exc2 2'
    finally:
        print 'finally'
    print 'done'

def baz(x):
    try:
        print 'try 1'
        square(x)
        print 'try 2'
    except Exc1 as e:
        print 'catch Exc1 1'
        print e.message
        print 'catch Exc1 2'
    except Exc2 as e:
        print 'catch Exc2 1'
        print e.message
        print 'catch Exc2 2'
    finally:
        print 'finally'
    print 'done'

def baz1(x):
    try:
        print 'try 1.1'
        foo(x)
        print 'try 1.2'
    except Exc1 as e:
        print 'catch Exc1 1.1'
        print e.message
        print 'catch Exc1 1.2'
    finally:
        print 'finally 1'
    print 'done 1'

def baz2(x):
    try:
        print 'try 2.1'
        baz1(x)
        print 'try 2.2'
    except:
        print 'catch Exc2'
    finally:
        print 'finally 2'
    print 'done 2'

def nest1(b):
    if b:
        raise C('C')
    else:
        raise E('E')

def nest2(b):
    try:
        try:
            try:
                try:
                    nest1(b)
                except A:
                    print 'A'
                finally:
                    print 'f A'
            except B:
                print 'B'
            finally:
                print 'f B'
        except C as c:
            print c.message
        finally:
            print 'f C'
    except D:
        print 'D'
    finally:
        print 'f D'

def nest3(b):
    try:
        nest2(b)
    except:
        print 'except'
    finally:
        print 'done'

def finally_return(x):
    try:
        try:
            return 'A'
        finally:
            if x < 5:
                return 'B'
    finally:
        if x > 10:
            return 'C'

def finally_return_void():
    try:
        try:
            print 'A'
            return
        finally:
            print 'B'
            return
    finally:
        print 'C'

def finally_break_continue1():
    try:
        for i in range(3):
            try:
                continue
            finally:
                print i
                continue
            print 'X'
    finally:
        print 'f'

def finally_break_continue2():
    try:
        for i in range(5):
            try:
                if i == 4:
                    continue

                for j in range(i):
                    try:
                        try:
                            if j == 3:
                                break
                            elif j == 1:
                                continue
                            print j
                        finally:
                            print 'f1'
                    finally:
                        print 'f2'
                        if j == 4:
                            break
            finally:
                print 'f3'
    finally:
        print 'f4'

def finally_break_continue3(n):
    while n != 0:
        print 'A'
        try:
            while n != 0:
                print 'B'
                if n == 42:
                    print 'C'
                    n -= 1
                    continue
                try:
                    print 'D'
                    if n > 0:
                        print 'E'
                        continue
                    else:
                        print 'F'
                        break
                finally:
                    print('G')
                    return -1
            print 'H'
            return -2
        finally:
            print 'I'
            return n + 1

def test_try_with_loop1(_):
    try:
        print 'A'
        while True:
            print 'B'
            try:
                print 'C'
                raise ValueError()
                assert False
            except ValueError:
                print 'D'
                break
    finally:
        print 'E'

def test_try_with_loop2(_):
    try:
        print 'A'
        while True:
            print 'B'
            try:
                print 'C'
                raise ValueError()
                assert False
            except:
                print 'D'
                break
    finally:
        print 'E'

def test_try_with_loop3(_):
    try:
        try:
            print 'A'
            while True:
                print 'B'
                try:
                    print 'C'
                    raise ValueError()
                    assert False
                except IOError:
                    print 'D'
                    break
        finally:
            print 'E'
    except:
        print 'F'
    finally:
        print 'G'

def test_try_with_loop4(_):
    try:
        try:
            print 'A'
            while True:
                print 'B'
                try:
                    print 'C'
                    raise ValueError()
                    assert False
                except:
                    print 'D'
                    raise IOError()
                    break
        finally:
            print 'E'
    except:
        print 'F'
    finally:
        print 'G'

# EXPECT: try
# EXPECT: catch Exc1 1
# EXPECT: e1
# EXPECT: catch Exc1 2
# EXPECT: finally
# EXPECT: done
bar(True)

# EXPECT: try
# EXPECT: catch Exc2 1
# EXPECT: e2
# EXPECT: catch Exc2 2
# EXPECT: finally
# EXPECT: done
bar(0)

# EXPECT: try 1
# EXPECT: try 2
# EXPECT: finally
# EXPECT: done
baz(3.14)

# EXPECT: try 2.1
# EXPECT: try 1.1
# EXPECT: catch Exc1 1.1
# EXPECT: e1
# EXPECT: catch Exc1 1.2
# EXPECT: finally 1
# EXPECT: done 1
# EXPECT: try 2.2
# EXPECT: finally 2
# EXPECT: done 2
baz2(1)

# EXPECT: try 2.1
# EXPECT: try 1.1
# EXPECT: finally 1
# EXPECT: catch Exc2
# EXPECT: finally 2
# EXPECT: done 2
baz2(0)

# EXPECT: f A
# EXPECT: f B
# EXPECT: C
# EXPECT: f C
# EXPECT: f D
# EXPECT: done
nest3(True)

# EXPECT: f A
# EXPECT: f B
# EXPECT: f C
# EXPECT: f D
# EXPECT: except
# EXPECT: done
nest3(0)

print finally_return(3.14)  # EXPECT: B
print finally_return(7)     # EXPECT: A
print finally_return(11)    # EXPECT: C

# EXPECT: A
# EXPECT: B
# EXPECT: C
finally_return_void()

# EXPECT: 0
# EXPECT: 1
# EXPECT: 2
# EXPECT: f
finally_break_continue1()

# EXPECT: f3
# EXPECT: 0
# EXPECT: f1
# EXPECT: f2
# EXPECT: f3
# EXPECT: 0
# EXPECT: f1
# EXPECT: f2
# EXPECT: f1
# EXPECT: f2
# EXPECT: f3
# EXPECT: 0
# EXPECT: f1
# EXPECT: f2
# EXPECT: f1
# EXPECT: f2
# EXPECT: 2
# EXPECT: f1
# EXPECT: f2
# EXPECT: f3
# EXPECT: f3
# EXPECT: f4
finally_break_continue2()

# EXPECT: A
# EXPECT: B
# EXPECT: C
# EXPECT: B
# EXPECT: D
# EXPECT: E
# EXPECT: G
# EXPECT: I
# EXPECT: 42
print finally_break_continue3(42)

# EXPECT: A
# EXPECT: B
# EXPECT: C
# EXPECT: D
# EXPECT: E
test_try_with_loop1(0)

# EXPECT: A
# EXPECT: B
# EXPECT: C
# EXPECT: D
# EXPECT: E
test_try_with_loop2(0.0)

# EXPECT: A
# EXPECT: B
# EXPECT: C
# EXPECT: E
# EXPECT: F
# EXPECT: G
test_try_with_loop3('')

# EXPECT: A
# EXPECT: B
# EXPECT: C
# EXPECT: D
# EXPECT: E
# EXPECT: F
# EXPECT: G
test_try_with_loop4(False)

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

def test_with():
    with Foo(0) as f:
        f.foo()
    with Foo(1) as f, Bar('s') as b:
        f.foo()
        b.bar()
    with Foo(2), Bar('t') as q:
        print 'eeh'
        q.bar()

# EXPECT: > foo! 0
# EXPECT: woof
# EXPECT: < foo! 0
# EXPECT: > foo! 1
# EXPECT: > bar! s
# EXPECT: woof
# EXPECT: meow
# EXPECT: < bar! s
# EXPECT: < foo! 1
# EXPECT: > foo! 2
# EXPECT: > bar! t
# EXPECT: eeh
# EXPECT: meow
# EXPECT: < bar! t
# EXPECT: < foo! 2
test_with()


class PropClass:
    x: int

    @property
    def foo(self: PropClass):
        raise IOError('foo')

def test_property_exceptions():
    try:
        PropClass(42).foo
        print 'X'
    except IOError as e:
        print e.message

# EXPECT: foo
test_property_exceptions()

def test_empty_raise():
    def foo(b):
        if b:
            raise ValueError('A')
        else:
            raise IOError('B')

    def bar(b):
        try:
            foo(b)
            print('X')
        except IOError as e:
            print(e)
            raise
        except:
            raise

    def baz(b):
        try:
            bar(b)
        except ValueError as e:
            print(e)
            raise

    for b in (False, True):
        try:
            baz(b)
        except:
            print('C')

# EXPECT: B
# EXPECT: C
# EXPECT: A
# EXPECT: C
test_empty_raise()
