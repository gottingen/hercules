.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _collections:

Collections
=======================

Collections we have two parts, the base collections are most closely same to the python collections, for mostly
compatibility with python code. The second part is the `EA` collections, implemented in C++ and optimized for performance.
in `turbo` and `flare` to using high performance data structures with `simd` and `GPU` support.

List, Dict, Set

.. code-block:: python

    l = [1, 2, 3]                         # type: List[int]; a list of integers
    s = {1.1, 3.3, 2.2, 3.3}              # type: Set[float]; a set of floats
    d = {1: 'hi', 2: 'ola', 3: 'zdravo'}  # type: Dict[int, str]; a dictionary of int to str

    ln = []                               # an empty list whose type is inferred based on usage
    ln = List[int]()                      # an empty list with explicit element type
    dn = {}                               # an empty dict whose type is inferred based on usage
    dn = Dict[int, float]()               # an empty dictionary with explicit element types
    sn = set()                            # an empty set whose type is inferred based on usage
    sn = Set[str]()                       # an empty set with explicit element type

Lists also take an optional capacity constructor argument, which can be useful when creating large lists:

.. code-block:: python

    l = List[int](capacity=1000)  # a list of integers with capacity for 1000 elements
    # add 1000 elements to the list
    for i in range(1000):
        l.append(i)

.. note::

    Standary Dictionaries and sets are unordered and are based on `klib <https://github.com/attractivechaos/klib>`_

Comprehensions
--------------------------
Comprehensions are a nifty, Pythonic way to create collections, and are fully supported:

.. code-block:: python

    l = [i for i in range(10)]  # type: List[int]; a list of integers from 0 to 9
    s = {i for i in range(10)}  # type: Set[int]; a set of integers from 0 to 9
    d = {i: i for i in range(10)}  # type: Dict[int, int]; a dictionary of int to int from 0 to 9

    # comprehensions can also be used with explicit types
    l = [i for i in range(10)]  # type: List[int]; a list of integers from 0 to 9
    s = {i for i in range(10)}  # type: Set[int]; a set of integers from 0 to 9
    d = {i: i for i in range(10)}  # type: Dict[int, int]; a dictionary of int to int from 0 to 9

You can also use generators to create collections:

.. code-block:: python

    g = (i for i in range(10))
    print(list(g))  # prints list of integers from 0 to 9, inclusive


Tuples
--------------------------

Tuples are immutable collections of elements, and are fully supported:

.. code-block:: python

    t = (1, 2, 3)  # type: Tuple[int, int, int]; a tuple of three integers
    t = Tuple[int, int, int](1, 2, 3)  # a tuple of three integers with explicit element types

    # tuples can also be nested
    t = ((1, 2), (3, 4))  # type: Tuple[Tuple[int, int], Tuple[int, int]]; a tuple of two tuples of integers

    # tuples can also be used with comprehensions
    t = tuple(i for i in range(10))  # type: Tuple[int, ...]; a tuple of integers from 0 to 9

As all types must be known at compile time, tuple indexing works only if a tuple is homogenous (all types are the same)
or if the value of the index is known at compile time.

You can, however, iterate over heterogenous tuples in hercules. This is achieved behind the scenes by unrolling the loop
to accommodate the different types.

.. code-block:: python

    t = (1, 'hi', 3.14)
    for i in t:
        print(i)

    # index
    x = int(some_dynamic_value_gen())
    t[x]  # compile error: x is not known at compile time

    t = (1, 2, 3)
    t[0]  # 1 ok, because the index is known at compile time

.. warning::

    Tuples are not hashable in hercules, and cannot be used as keys in dictionaries.
    Tuple are immutable, so `t = (1, 2, 3); t[0] = 4` will raise a compile error.

tuples supports most of Python's tuple unpacking syntax:

.. code-block:: python

    x, y = 1, 2                # x is 1, y is 2
    (x, (y, z)) = 1, (2, 3)    # x is 1, y is 2, z is 3
    [x, (y, z)] = (1, [2, 3])  # x is 1, y is 2, z is 3

    l = range(1, 8)    # l is [1, 2, 3, 4, 5, 6, 7]
    a, b, *mid, c = l  # a is 1, b is 2, mid is [3, 4, 5, 6], c is 7
    a, *end = l        # a is 1, end is [2, 3, 4, 5, 6, 7]
    *beg, c = l        # c is 7, beg is [1, 2, 3, 4, 5, 6]
    (*x, ) = range(3)  # x is [0, 1, 2]
    *x = range(3)      # error: this does not work

    *sth, a, b = (1, 2, 3, 4)      # sth is (1, 2), a is 3, b is 4
    *sth, a, b = (1.1, 2, 3.3, 4)  # error: this only works on homogenous tuples for now

    (x, y), *pff, z = [1, 2], 'this'
    print(x, y, pff, z)               # x is 1, y is 2, pff is an empty tuple --- () ---, and z is "this"

    s, *q = 'XYZ'  # works on strings as well; s is "X" and q is "YZ"

Strong typing
--------------------------

Because hercules is strongly typed, these won't compile:

.. code-block:: python

    l = [1, 's']   # is it a List[int] or List[str]? you cannot mix-and-match types
    d = {1: 'hi'}
    d[2] = 3       # d is a Dict[int, str]; the assigned value must be a str

    t = (1, 2.2)  # Tuple[int, float]
    lt = list(t)  # compile error: t is not homogenous

    lp = [1, 2.1, 3, 5]  # compile error: Hercules will not automatically cast a float to an int


it works  like this:

.. code-block:: python

    u = (1, 2, 3)
    lu = list(u)  # works: u is homogenous