.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _classes:

Classes
====================
Hercules supports classes just like Python. However, you must declare class members and their types in the preamble
of each class (like you would do with Python's dataclasses):

.. code-block::

    class Foo:
    x: int
    y: int

    def __init__(self, x: int, y: int):  # constructor
        self.x, self.y = x, y

    def method(self):
        print(self.x, self.y)

    f = Foo(1, 2)
    f.method()  # prints "1 2"

Unlike Python, Hercules supports method overloading:

.. code-block::

    class Foo:
    x: int
    y: int

    def __init__(self, x: int, y: int):  # constructor
        self.x, self.y = x, y

    def __init__(self, x: int):  # constructor overload
        self.x, self.y = x, 0

    def method(self):
        print(self.x, self.y)

    f = Foo(1)
    f.method()  # prints "1 0"
    f = Foo(1, 2)
    f.method()  # prints "1 2"
    f = Foo()  # error: missing argument
    f = Foo(1, 2, 3)  # error: too many arguments
    f = Foo(1, "2")  # error: wrong type
    f = Foo(1.1, 2)  # error: wrong type

generic
--------------------
Hercules also supports generic classes, like C++ templates:

.. code-block::

    class Foo[T]:
        x: T
        y: T
        def __init__(self, x: T, y: T):
            self.x, self.y = x, y

        def method(self):
            print(self.x, self.y)
    f = Foo(1, 2)

when you declare a generic class, you need take care of the type of the arguments you pass to the class,
if you pass a wrong type, Hercules will raise an error. As in C/C++, some types are compatible with others, for example,
you can pass an int to a float, but you can't pass a float to an int.

.. code-block::

    f = Foo(1, 2)  # ok
    f = Foo(1.1, 2.2)  # ok
    f = Foo(1, 2.2)  # error: wrong type
    f = Foo(1.1, 2)  # ok (int is compatible with float)

References
--------------------

In hercules, we always try to avoid copying data, so when you pass a class assigment to another class of the same type,
you are passing a reference to the original class, not a copy of it.

.. code-block::

    class Foo:
        x: int
        y: int

        def __init__(self, x: int, y: int):  # constructor
            self.x, self.y = x, y

        def method(self):
            print(self.x, self.y)

    f = Foo(1, 2)
    g = f  # g is a reference to f
    g.x = 3
    f.method()  # prints "3 2"

If you need to copy an object's contents, implement the __copy__ magic method and use q = copy(p) instead.

.. code-block::

    class Foo:
        x: int
        y: int

        def __init__(self, x: int, y: int):  # constructor
            self.x, self.y = x, y

        def __copy__(self):
            return Foo(self.x, self.y)

        def method(self):
            print(self.x, self.y)

    f = Foo(1, 2)
    g = copy(f)  # g is a copy of f
    g.x = 3
    f.method()  # prints "1 2"

Inheritance
--------------------


Classes can inherit from other classes:

.. code-block::

    class Foo:
        x: int
        y: int

        def __init__(self, x: int, y: int):  # constructor
            self.x, self.y = x, y

        def method(self):
            print(self.x, self.y)

    class Bar(Foo):
        z: int

        def __init__(self, x: int, y: int, z: int):  # constructor
            super().__init__(x, y)
            self.z = z

        def method(self):
            super().method()
            print(self.z)

    b = Bar(1, 2, 3)
    b.method()  # prints "1 2 3"

.. note::

    Currently, inheritance in Hercules is still under active development. Treat it as a beta feature.

Named tuples
--------------------

Hercules also supports pass-by-value types via the @tuple annotation, which are effectively named tuples
(equivalent to Python's collections.namedtuple):

.. code-block::

    @tuple
    class Point:
        x: int
        y: int

    p = Point(1, 2)
    print(p.x, p.y)  # prints "1 2"

However, named tuples are immutable. The following code will not compile:

.. code-block::

    p.x = 3  # error: named tuples are immutable

You can also add methods to named tuples:

.. code-block::

    @tuple
    class Point:
        x: int
        y: int

        def method(self):
            print(self.x, self.y)

    p = Point(1, 2)
    p.method()  # prints "1 2"

static class methods
-----------------------

Hercules supports static class methods, which are methods that do not require an instance of the class to be called, we define
a static method using the @staticmethod annotation:

.. code-block::

    class Foo:
        x: int
        y: int

        def __init__(self, x: int, y: int):  # constructor
            self.x, self.y = x, y

        @staticmethod
        def static_method():
            print("static method")

    Foo.static_method()  # prints "static method"

also, static methods can be overloaded, as in the following example:

.. code-block::

    class Foo:
        x: int
        y: int

        def __init__(self, x: int, y: int):  # constructor
            self.x, self.y = x, y

        @staticmethod
        def static_method():
            print("static method")

        @staticmethod
        def static_method(x: int):
            print("static method with argument", x)

    Foo.static_method()  # prints "static method"
    Foo.static_method(1)  # prints "static method with argument 1"

static class variables
----------------------------

Hercules also supports static class variables, which are variables that are shared among all instances of a class, we define
a static variable using the ClassVar annotation:

.. code-block::

    class Foo:
        z: ClassVar[int] = 0
    print(Foo.z)
    Foo.z = 10
    print(Foo.z)

Type extensions
--------------------

Suppose you have a class that lacks a method or an operator that might be really useful. Hercules provides an @extend
annotation that allows programmers to add and modify methods of various types at compile time, including built-in types
like int or str. This actually allows much of the functionality of built-in types to be implemented in Hercules as type
extensions in the standard library.

.. code-block::

    @extend
    class Point:
        def euclidean_distance(self, other: Point) -> float:
            return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

Magic methods
--------------------

Here is a list of useful magic methods that you can implement in your classes:

.. list-table::
    :header-rows: 1
    :widths: 20 50

    * - `Magic method`
      - `Description`
    * - `__init__`
      - `Constructor`
    * - `__copy__`
      - `Copy constructor`
    * - `__len__`
      - `for len(obj)`
    * - `__bool__`
      - `for bool method and condition checking`
    * - `__getitem__`
      - `overload obj[key]`
    * - `__setitem__`
      - `overload obj[key] = value`
    * - `__delitem__`
      - `overload del obj[key]`
    * - `__iter__`
      - `support iterating over the object`
    * - `__repr__`
      - `support printing and str conversion`

magic methods are a way to overload operators in Python, and Hercules supports most of them. For a complete list of magic
methods, see the Python documentation for more information.


