
@trace_debug
def foo(x):
    return x*3 + x

def validate(x, y):
    print(x, y)
    assert y == x*4

a = foo(10)
b = foo(1.5)
c = foo('a')
