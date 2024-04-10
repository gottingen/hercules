.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _types:

Types and features
=========================

we supports a number of additional types that are not present in plain Python.

Arbitrary-width foundational types
------------------------------------------

Commonly int type is a 64-bit signed integer. However, Hercules supports arbitrary-width signed and unsigned integers:

.. code-block:: python

    a = Int[16](42)    # signed 16-bit integer 42
    b = UInt[128](99)  # unsigned 128-bit integer 99

standard library provides shorthands for the common variants:

..  list-table::
    :widths: 20 80
    :header-rows: 1

    * - Type
      - Description
    * - `i8`
      - 8-bit signed integer
    * - `i16`
      - 16-bit signed integer
    * - `i32`
      - 32-bit signed integer
    * - `i64`
      - 64-bit signed integer
    * - `u8`
      - 8-bit unsigned integer
    * - `u16`
      - 16-bit unsigned integer
    * - `u32`
      - 32-bit unsigned integer
    * - `u64`
      - 64-bit unsigned integer
    * - `f32`
      - 32-bit floating point number
    * - `f64`
      - 64-bit floating point number

Pointers
------------------------------------------

Pointers are a common feature in many programming languages. They are used to store the memory address of another value.
the keyword `Ptr` is used to declare a pointer type.as a `Ptr[T]` type that represents a pointer to an object of type `T`
Pointers can be useful when interfacing with `C`. The `__ptr__` keyword can also be used to obtain a pointer to a variable:

.. code-block:: python

    a = Ptr[i32]() # pointer to a 32-bit signed integer, initialized to NULL
    b = Ptr[i32](42) # pointer to a 32-bit signed integer, initialized to allocated memory containing 42

    c = 42
    d = c.__ptr__() # pointer to the integer c

    from C import some_function(Ptr[i32]) # declare a function that takes a pointer to a 32-bit signed integer
    some_function(d) # pass the pointer to the function

The `cobj` alias corresponds to `void*` in C and represents a generic C or C++ object.

for example, the `C` library function `malloc` returns a `void*` pointer to allocated memory.
The `Ptr` type can be used to store this pointer and access the allocated memory:

.. code-block:: python

    from C import malloc(i32) -> cobj # declare the malloc function
    a = malloc(4) # allocate 4 bytes of memory and store the pointer in a
    b = Ptr[i32](a)
    if b:
        print(b[0]) # access the first 32-bit integer in the allocated memory
     else:
        raise Exception("malloc failed")


.. note::

    Using pointers directly circumvents any runtime checks, so dereferencing a null pointer, for example,
    will cause a segmentation fault just like in C/C++.

    It is also as a good practice to always check if a pointer is null before dereferencing it.

Static arrays
------------------------------------------

The `__array__` keyword can be used to allocate static arrays on the stack:

.. code-block:: python

    def foo(n):
    arr = __array__[int](5)  # similar to "long arr[5]" in C
    arr[0] = 11
    arr[1] = arr[0] + 1
    ...

