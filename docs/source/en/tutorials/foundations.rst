.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _foundation:

Foundation
=====================

Hercules is designed to `write python code` that is so easy to convert to other C/C++ languages which are more efficient
and faster than python. The main goal is to write python code that is easy to convert to C/C++ code. So, if you are
familiar with python, you already be a master of 95% of Hercules.

In this section, all the code examples are written in jupyter notebook. You can find the jupyter notebook in the
`examples/tutorial/foundations.ipynb` file.

Print
---------------------

.. code-block:: python

    print("Hello, World!")

.. code-block:: python

    from sys import stderr
    print('hello world', end='', file=stderr)


Comments
---------------------

.. code-block:: python

    # This is a comment

    """
    Multi-line comments are
    possible like this.
    """

Literals
---------------------

.. code-block:: python

    # Booleans
    True   # type: bool
    False

    # Numbers
    a = 1             # type: int; a signed 64-bit integer
    b = 1.12          # type: float; a 64-bit float (just like "double" in C)
    c = 5u            # unsigned int; an unsigned 64-bit int
    d = Int[8](12)    # 8-bit signed integer; you can go all the way to Int[2048]
    e = UInt[8](200)  # 8-bit unsigned integer
    f = byte(3)       # Hercules's byte is equivalent to C's char; equivalent to Int[8]

    h = 0x12AF   # hexadecimal integers are also welcome
    g = 3.11e+9  # scientific notation is also supported
    g = .223     # and this is also float
    g = .11E-1   # and this as well

    # Strings
    s = 'hello! "^_^" '              # type: str
    t = "hello there! \t \\ '^_^' "  # \t is a tab character; \\ stands for \
    raw = r"hello\n"                 # raw strings do not escape slashes; this would print "hello\n"
    fstr = f"a is {a + 1}"           # an f-string; prints "a is 2"
    fstr = f"hi! {a+1=}"             # an f-string; prints "hi! a+1=2"
    t = """
    hello!
    multiline string
    """

    # The following escape sequences are supported:
    #   \\, \', \", \a, \b, \f, \n, \r, \t, \v,
    #   \xHHH (HHH is hex code), \OOO (OOO is octal code)


Assignments and operators
-----------------------------------

.. code-block:: python

    a = 1 + 2              # this is 3
    a = (1).__add__(2)     # you can use a function call instead of an operator; this is also 3
    a = int.__add__(1, 2)  # this is equivalent to the previous line
    b = 5 / 2.0            # this is 2.5
    c = 5 // 2             # this is 2; // is an integer division
    a *= 2                 # a is now 6

Each operator has a corresponding magic method that can be called explicitly. For example, `+` is `__add__`, `-` is
`__sub__`.below is the list of binary operators and each one's associated magic method:

.. list-table::
    :widths: 20 30
    :header-rows: 1

    * - `Operator`
      - `Magic Method`
    * - `+`
      - `__add__`
    * - `-`
      - `__sub__`
    * - `*`
      - `__mul__`
    * - `/`
      - `__truediv__`
    * - `//`
      - `__floordiv__`
    * - `%`
      - `__mod__`
    * - `**`
      - `__pow__`
    * - `@`
      - `__matmul__`
    * - `<<`
      - `__lshift__`
    * - `>>`
      - `__rshift__`
    * - `&`
      - `__and__`
    * - `|`
      - `__or__`
    * - `^`
      - `__xor__`
    * - `<<`
      - `__lshift__`
    * - `>>`
      - `__rshift__`
    * - `<`
      - `__lt__`
    * - `<=`
      - `__le__`
    * - `==`
      - `__eq__`
    * - `!=`
      - `__ne__`
    * - `in`
      - `__contains__`
    * - `and`
      - `n/a`
    * - `or`
      - `n/a`


there are also unary operators, expressions like `b = +a`, `-a`.

.. list-table::
    :widths: 20 30 50
    :header-rows: 1

    * - `Operator`
      - `Magic Method`
      - `Description`
    * - `+`
      - `__pos__`
      - `unary positive`
    * - `-`
      - `__neg__`
      - `unary negative`
    * - `~`
      - `__invert__`
      - `bitwise NOT`


Control flow
---------------------

If statements
_____________________

common if statement like in python:

.. code-block:: python

    d = 2
    if d > 3:
        print("d is greater than 3")
    elif d == 3:
        print("d is equal to 3")
    else:
        print("d is less than 3")

extended c++'s switch statement to be a `match` statement:

.. code-block:: python

    def random_number():
        import random
        return random.randint(0, 20)

    def match_ex():
        for _ in range(20):
            match random_number():  # assuming that the type of this expression is int
                case 1:         # is it 1?
                    print('hi')
                case 2 ... 10:  # is it 2, 3, 4, 5, 6, 7, 8, 9 or 10?
                    print('wow!')
                case _:         # "default" case
                    print('meh...')

    match_ex()

The `match` statement is a powerful tool that can be used to replace `if-elif-else` statements. It can be used to

.. code-block:: python

    match bool_expr():  # now it's a bool expression
        case True:
            print('yay')
        case False:
            print('nay')

The `match` statement can also be used to match against other types of expressions:

.. code-block:: python

    match str_expr():  # now it's a str expression
        case 'abc': print("it's ABC time!")
        case 'def' | 'ghi':  # you can chain multiple rules with the "|" operator
            print("it's not ABC time!")
        case s if len(s) > 10: print("so looong!")  # conditional match expression
        case _: assert False

    match some_tuple:  # assuming type of some_tuple is Tuple[int, int]
        case (1, 2): ...
        case (a, _) if a == 42:  # you can do away with useless terms with an underscore
            print('hitchhiker!')
        case (a, 50 ... 100) | (10 ... 20, b):  # you can nest match expressions
            print('complex!')

list matching:

.. code-block:: python

    match list_foo():
        case []:                   # [] matches an empty list
            print('A')
        case [1, 2, 3]:            # make sure that list_foo() returns List[int] though!
            print('B')
        case [1, 2, ..., 5]:       # matches any list that starts with 1 and 2 and ends with 5
            print('C')
        case [..., 6] | [6, ...]:  # matches a list that starts or ends with 6
            print('D')
        case [..., w] if w < 0:    # matches a list that ends with a negative integer
            print('E')
        case [...]:                # any other list
            print('F')

Loops
________________________

hercules supports the standard python loops like `for` and `while` loops.

for loop, `for` construct an iterator over any generator, which means any object that has a `__iter__` method. In Practice,
this means that generators, lists, tuples, sets, and homogenous tuples, ranges, and many more types implement this method.
If you need to implement one yourself, just keep in mind that __iter__ is a generator and not a function.

.. code-block:: python

    for i in range(5):
        print(i)

    for i in range(1, 10, 2):  # start, stop, step
        print(i)

    for i in range(10, 1, -1):  # start, stop, step
        print(i)

    for i in [1, 2, 3, 4, 5]:
        print(i)

    for i in 'hello':
        print(i)

    for i in range(5):
        if i == 3:
            break
            print(i)
        else:
            print('no break')

while loop:

.. code-block:: python

    i = 0
    while i < 5:
        print(i)
        i += 1

    i = 0
    while True:
        if i == 5:
            break
        print(i)
        i += 1

    i = 0
    while i < 5:
        i += 1
        if i == 3:
            continue
        print(i)


Imports
---------------------

The `import` statement is used to import modules. The `from` statement is used to import specific functions or classes from
a module.

like below, importing the system module `math` and using the `sqrt` function:

.. code-block:: python

    import math
    print(math.sqrt(4))

    from math import sqrt
    print(sqrt(4))

importing a module with an alias:

.. code-block:: python

    import math as m
    print(m.sqrt(4))

    from math import sqrt as s
    print(s(4))

using the `from` statement to import all functions from a module:

.. code-block:: python

    from math import *
    print(sqrt(4))

    from math import sqrt, sin, cos
    print(sqrt(4))
    print(sin(0))
    print(cos(0))

    from math import sqrt as s, sin as si, cos as c
    print(s(4))
    print(si(0))
    print(c(0))

importing a `C` function. The `C` function is a function that is written in `C` and can be used in `Hercules`,
all the `C` functions are defined in the `C` module.

Assuming that you defined a function `print_hello` in a file called `print_hello.c`:

.. code-block:: c

    #include <stdio.h>

    void print_hello() {
        printf("Hello, World!\n");
    }

and you compiled it to a shared library called `print_hello.so`:

.. code-block:: bash

    gcc -shared -o print_hello.so print_hello.c

then you can import and use it in Hercules like this:

.. code-block:: python

    from C import print_hello
    print_hello()

then `import some_module` looks for a file called `some_module.hs` in the current directory and stdlib directories, or
looks for a directory called `some_module` and tries to import `some_module/__init__.hs`.

.. note::

    The `C` module is a special module that is used to import `C` functions. But it does not specify the path of the
    dynamic library and the library should be loaded in the current app context. Next mailstone design to specify the
    name of the dynamic library and the path of the dynamic library, and load the dynamic library in the current app
    context. it is assumed like this:

    `from C(path='path/to/dynamic/library', name='dynamic_library_name') import function_name`.`

    the path can is optional and if it is not specified, and can be relative to the `rpath` of the current app context, or
    the absolute path.

    the name is optional and if it is not specified, it is assumed to load the function from the dynamic libraries that has
    loaded in the current app context.


Exceptions
---------------------

hercules supports the standard python exceptions like `try`, `except`, `finally` and `raise`. below is an example of
how to use them:

.. code-block:: python

    def throwable():
     raise ValueError("doom and gloom")

    try:
        throwable()
    except ValueError as e:
        print("we caught the exception")
    except:
        print("ouch, we're in deep trouble")
    finally:
        print("whatever, it's done")

.. warning::

    Right now, Hercules cannot catch multiple exceptions in one statement. Thus catch (Exc1, Exc2, Exc3) as var
    will not compile, since the type of var needs to be known ahead of time.

Context life cycle
---------------------

If you have an object that implements `__enter__` and `__exit__` methods to manage its lifetime (say, a `File`),
you can use a `with` statement to make your life easier:

.. code-block:: python

    with open('file.txt', 'w') as f:
        f.write('hello, world!')

    # f is closed here

Statics
---------------------

Sometimes, certain values or conditions need to be known at compile time. For example, the bit width `N` of an integer
type `Int[N]`, or the size `M` of a static array `__array__[int](M)` need to be compile time constants.

To accomodate this, `Hercules` uses `static values`, i.e. values that are known and can be operated on at compile time.
`Static[T]` represents a static value of type `T`. Currently, `T` can only be `int` or `str`.

For example, we can parameterize the bit width of an integer type as follows:

.. code-block:: python

    N: Static[int] = 32

    a = Int[N](10)      # 32-bit integer 10
    b = Int[2 * N](20)  # 64-bit integer 20

All of the standard arithmetic operations can be applied to static integers to produce new static integers.

Statics can also be passed to the compiler via the -D flag, as in -DN=32.

Classes can also be parameterized by statics:

.. code-block:: python

    class MyInt[N: Static[int]]:
    n: Int[N]

    x = MyInt[16](i16(42))

this equivalent to the following code in C++:

.. code-block:: c++

    template<int N>
    class MyInt {
    public:
        int n[N];
    };

    MyInt<16> x;

.. note::


Static evaluation
---------------------

In certain cases a program might need to check a particular type and perform different actions
based on it. For example:

.. code-block:: python

    def flatten(x):
    if isinstance(x, list):
        for a in x:
            flatten(a)
    else:
        print(x)

    flatten([[1,2,3], [], [4, 5], [6]])

Standard static typing on this program would be problematic since, if x is an int, it would not be iterable and
hence would produce an error on for a in x. we solves this problem by evaluating certain conditions at compile time,
such as isinstance(x, list), and avoiding type checking blocks that it knows will never be reached. In fact, this
program works and flattens the argument list.

Static evaluation works with plain static types as well as general types used in conjunction with type,
`isinstance` or `hasattr`.

this may like c++11's type traits, but it is more powerful and can be used in more cases.
like the following example:

.. code-block::

    template<typename T>
    struct is_vector : std::false_type {};

    template<typename T>
    struct is_vector<std::vector<T>> : std::true_type {};

    template<typename T>
    void some_function( const T& t ) {
        if constexpr( is_vector<T>::value ) {
            // do something with a vector
        } else {
            // do something with a non-vector
        }
    }

.. note::

    The `if constexpr` statement is a C++17 feature that allows you to conditionally compile code based on a compile-time
    constant. It is similar to the `if` statement, but the condition is evaluated at compile time, and the branch that
    is not taken is discarded from the compiled binary.

.. note::

    The C++17 `if constexpr` statement  and hercules's static evaluation are similar, but the hercules's static evaluation
    is more powerful and can be used in more cases. They all inspired by the functional programming languages like erlang's
    pattern matching. C++ and hercules's static evaluation are more efficient than python's solution, because they are
    evaluated at compile time, and provide a chance to optimize the code in IR pass.