.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _functions:

Functions
=============

like pythons functions, define a function with the `def` keyword. as follows:
A function returns a value with the `return` keyword.

.. code-block:: python

    def add(a, b):
        return a + b

    print(add(1, 2))

    # output: 3
    3


non return functions

.. code-block:: python

    def say_hello():
        print("Hello")

    say_hello()

    # output: Hello
    Hello


Restrict function Arguments and Return Types
---------------------------------------------
If we want to compile to a library or import the function that implements in C/C++ library or in future we want to
use the function definition in C/C++ code, we can restrict the function arguments and return types. Therefore, hercules
supports strong typing function arguments and return types.

define a function with strict typing arguments:

.. code-block:: python

    def add(a: int, b: int) -> int:
        return a + b

    print(add(1, 2))

    # output: 3
    3
as you can see, the function `add` takes two arguments `a` and `b` and both arguments are integers. The function returns
an integer value. The Argument type is defined with a colon `:` after the argument name. The return type is defined with
an arrow `->` before return type ,and the return type is followed by the colon `:` for function definition. The function
body is defined with an indentation.

the return value type can be Omitted, if the function does not return any value. the same semantics as the return value
`-> None`, as  the semantics of the return value type `void` in C/C++.
see a example:

.. code-block:: python

    def say_hello() -> None:
        print("Hello")

    say_hello()

    # output: Hello
    Hello

Definition of the same semantics:

.. code-block:: python

    def say_hello():
        print("Hello")


Definition of the same semantics defined in C/C++:

.. code-block:: c

    void say_hello() {
        printf("Hello");
    }

At this point, this allows us to transform the definitions of hs, python, and C/C++ into a unified AST
abstraction, and uniformly optimize and transform each other. Most importantly, it allows us to define
a function in python and use it in C/C++ code, or define a function in C/C++ code and use it in python code.
More information about the AST and IR transformation will be discussed in the `advance` section.

see the definition of the function foo:

.. code-block:: python

    def foo(a: int, b: int) -> int:
        return a + b

    print(foo(1, 2))

    # output: 3
    3
    print(foo(1.0, 2.0))  # error message: TypeError: foo() argument 1 must be int, got float
    print(foo("1", "2"))  # error message: TypeError: foo() argument 1 must be int, got str
    print(foo(1, 2, 3))  # error message: TypeError: foo() takes 2 positional arguments but 3 were given
    print(foo())  # error message: TypeError: foo() missing 2 required positional arguments: 'a' and 'b'
    print(foo(1.1, 2))  # error message: TypeError: foo() argument 1 must be int, got float


partial specialization of the function arguments:
---------------------------------------------------------------

We can define a function with a partial specialization of the function arguments. and do not specify the type of the
return value. see the example:

.. code-block:: python

    def bar(a: int, b) :
        return a + b

    print(bar(1, 2)) # output: 3
    print(bar(1, 2.0)) # output: 3.0 works for int implement of __add__(float)
    print(bar(1.0, 2)) # error message: TypeError: bar() argument 1 must be int, got float
    print(bar(1, "2")) # error message: TypeError: bar() argument 1 must be int, got str


we also can define a function with non type specialization of the function arguments, but with the return value type.
see the example:

.. code-block:: python

    def baz(a, b) -> int:
        return a + b

    print(baz(1, 2)) # output: 3
    print(baz(1, 2.0)) # error message: TypeError: baz() argument 2 must be int, got float
    print(baz(1.0, 2)) # error message: TypeError: baz() argument 1 must be int, got float
    print(baz(1, "2")) # error message: TypeError: baz() argument 1 must be int, got str


Default and named arguments
--------------------------------------------

We can define a function with default arguments. see the example:

.. code-block:: python

    def add(a: int, b: int = 1) -> int:
        return a + b

    print(add(1)) # output: 2
    print(add(1, 2)) # output: 3

function with named arguments:

.. code-block:: python

    def bas(a: int, b: int, c: str) -> str:
        return f"{a} + {b} = {c}"

    print(bas(1, 2, "3")) # output: 1 + 2 = 3
    print(bas(1, c="3", 2)) # output: 1 + 2 = 3

Optional Arguments
------------------

We can define a function with optional arguments. see the example:

.. code-block:: python

    def add(a: int, b: Optional[int] = None) -> int:
        if b is None:
            return a
        return a + b

    print(add(1)) # output: 1
    print(add(1, 2)) # output: 3

    print(add(1, None)) # output: 1
    print(add(1, "2")) # error message: TypeError: add() argument 2 must be int, got str

implicit Optional Arguments
-----------------------------

We can define a function with implicit optional arguments. see the example:

.. code-block:: python

    def add(a: int, b: int = None) -> int:
        if b is None:
            return a
        return a + b

At this case, the function `add` is equivalent to the function `add` as above promote argument `b` to optional arguments.

Generics
----------------

We emulates Python's lax runtime type checking using a technique known as monomorphization. If a function has an
argument without a type definition, we will treat it as a generic function, and will generate different instantiations
for each different invocation:

.. code-block:: python

    def foo(x):
        print(x)  # print relies on typeof(x).__repr__(x) method to print the representation of x

    foo(1)        # automatically generates foo(x: int) and calls int.__repr__ when needed
    foo('s')      # automatically generates foo(x: str) and calls str.__repr__ when needed
    foo([1, 2])   # automatically generates foo(x: List[int]) and calls List[int].__repr__ when needed

But what if you need to mix type definitions and generic types? Say, your function can take a list of anything? You can
use generic type parameters:

.. code-block:: python

    def foo(x: List[T]):
        print(x)  # print relies on typeof(x).__repr__(x) method to print the representation of x

    foo([1, 2])   # automatically generates foo(x: List[int]) and calls List[int].__repr__ when needed
    foo(['s'])    # automatically generates foo(x: List[str]) and calls List[str].__repr__ when needed

Generators
----------------

Hercules supports generators, and in fact they are heavily optimized in the compiler so as to typically eliminate
any overhead:

.. code-block:: python

    def gen(n):
    i = 0
    while i < n:
        yield i ** 2
        i += 1

    print(list(gen(10)))  # prints [0, 1, 4, ..., 81]
    print(list(gen(0)))   # prints []


You can also use yield to implement coroutines: yield suspends the function, while (yield) (i.e. with parenthesis)
receives a value, as in Python.

.. code-block:: python

    def mysum(start):
    m = start
    while True:
        a = (yield)     # receives the input of coroutine.send() call
        if a == -1:
            break       # exits the coroutine
        m += a
    yield m

    iadder = mysum(0)       # assign a coroutine
    next(iadder)            # activate it
    for i in range(10):
        iadder.send(i)      # send a value to coroutine
    print(iadder.send(-1))  # prints 45

Generator expressions are also supported:

.. code-block:: python

    squares = (i ** 2 for i in range(10))
    for i,s in enumerate(squares):
        print(i, 'x', i, '=', s)