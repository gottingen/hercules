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
    f = byte(3)       # Codon's byte is equivalent to C's char; equivalent to Int[8]

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

Control flow
---------------------

Loops
---------------------

Imports
---------------------

Exceptions
---------------------

Statics
---------------------

