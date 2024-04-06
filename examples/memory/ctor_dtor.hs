
class Foo:
    i: int
    def __enter__(self: Foo):
        print 'enter foo! ' + str(self.i)
    def __exit__(self: Foo):
        print 'exit foo! ' + str(self.i)
    def foo(self: Foo):
        print 'woof'


with Foo(0) as f:
    #: > foo! 0
    f.foo()  #: woof