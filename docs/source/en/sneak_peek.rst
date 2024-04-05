.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _sneak-peek:

Sneak Peek
========================

before we dive into the details, here is a sneak peek of what we are going to, and how hercules can help you.
If you are already familiar with python, you can start using hercules right away with nothing more than a few lines of code.
Even if you only have one finger that can press the keyboard, don’t be afraid, play Hercules smoothly with me.

hello world
------------------------

Now let's start with a simple example, the classic "Hello, World!" program.
code location: `examples/peek/hello_world.hs`

.. code-block:: python

    print("hello, hercules, I love you more than a Hamburger!")

.. code-block:: bash

    $ hercules hello_world.hs
    hello, hercules, I love you more than a Hamburger!

Congratulations! You have successfully run your first Hercules program.

run jit
------------------------

Now let's try a more complex example, the classic "JIT" program.
Use jit mode to run the program that we just wrote.

..  code-block:: bash

    $ hercules jit hello_world.hs
    >>> Hercules JIT v0.2.6 <<<
    hello, hercules, I love you more than a Hamburger!
    [done]

a run a bit more quickly, right?

build to binary
------------------------

Now let's try a more amazing example, the classic "Build to Binary" program.
using the `hello_world.hs` file, we can build it to a binary file.

..  code-block:: bash

    $ hercules build hello_world.hs
    $ ll
    drwxrwxr-x 2 zzz zzz 4.0K x月   y 07:23 ./
    drwxrwxr-x 3 zzz zzz 4.0K x月   y 06:46 ../
    -rwxrwxr-x 1 zzz zzz  80K x月   y 07:23 hello_world*
    -rw-rw-r-- 1 zzz zzz   60 x月   y 06:47 hello_world.hs
    $ ./hello_world
    hello, hercules, I love you more than a Hamburger!

excellent! you have successfully built your first Hercules program to a binary file.
and you can run it directly.

There is another example in the `examples/peek` args.hs, you can try it by yourself.

build to shared library
-------------------------------

code location: `examples/peek/add.hs` and `examples/peek/call_add.hs`

Now let's try a more useful example, the classic "Build to Shared Library" program.

First, let's define a function name `simple_add` in the `add.hs` file.  and call it in the `call_add.hs` file.

`add.hs`:

.. code-block:: python

    def simple_add(a:int, b:int) -> int:
        return a + b



`call_add.hs`:

.. code-block:: python

    from add import simple_add

    print(simple_add(3,2))

now we run it:

.. code-block:: bash

    $ hercules run  call_add.hs
    $ 5

take attention, next, we will build the `simple_add` to a shared library.In order to compile it
into a dynamic library, we need to do a little extra work. For better comparison, we define the exported function as
`shared_add` for the same function.

add_shared.hs, we add these:

.. code-block:: python

    @export
    def shared_add(a:int, b:int) -> int:
        return a + b

Yes, you see, we add `@export` for the `shared_add`. there are two points we need to care.

* Function parameters and return value must specify explicit types
* must be mark attribute `@export`

build library:

.. code-block:: bash

    $ hercules build -o libadd_shared.so -r pic add.hs

you will see `libshared_add.so` in the dir. let us see the symbol it export


.. code-block:: bash

    $ nm libadd_shared.so |grep add
    00000000000031f0 T int:int.__add__:1[int,int].10
    00000000000038a0 T shared_add

so, you can see that, we export `shared_add` function and function `simple_add` is not be export.

We have prepared the dynamic library, next sections, will call the function from `hs`, `C` and `C++`.

call from hs
-------------------------------

call_add.hs

.. code-block:: python

    LIB="./libadd_shared.so"
    from C import LIB.shared_add(int, int)->int
    print(shared_add(3,2))

run by hercules

.. code-block:: bash

    $ hercules run call_add.hs
    5
    5

We can see that the function `shared_add` is called successfully.

call from C
-------------------------------

code location: `examples/peek/call_add.c`

call_add.c:

.. code-block:: c

    #include <stdio.h>

    extern int shared_add(int, int);

    int main() {
        int a = 2;
        int b = 3;
        int c = shared_add(a, b);
        printf("The sum of %d and %d is %d\n", a, b, c);
        return 0;
    }

compile and run:

.. code-block:: bash

    $ gcc -o call_add call_add.c -L. -ladd_shared -Wl,-rpath=.
    $ ./call_add
    The sum of 2 and 3 is 5

We can see that the function `shared_add` is called successfully.

.. note::

    In C/C++ mode, we need to add `-L. -ladd_shared` to the compile command to link the shared library.
    * Add the rpath to the gcc/g++ command: -Wl,-rpath=/path/to/lib/dir
    * Add the rpath to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=/path/to/lib/dir:$LD_LIBRARY_PATH


call from C++
-------------------------------

.. code-block:: c++

    #include <stdio.h>

    extern "C" int shared_add(int, int);

    int main() {
        int a = 2;
        int b = 3;
        int c = shared_add(a, b);
        printf("The sum of %d and %d is %d\n", a, b, c);
        return 0;
    }

compile and run:

.. code-block:: bash

    $ g++ -o call_add_cc call_add.cc -L. -ladd_shared -Wl,-rpath=.
    $ ./call_add_cc
    The sum of 2 and 3 is 5

Congratulations! You have successfully called the shared library function from `hs`, `C` and `C++`.

.. note::

    There are a little difference between `hs`, `C` and `C++` when calling the shared library function.
    In `hs`, we use `from C import LIB.shared_add(int, int)->int` to import the function.
    In `C`, we use `extern int shared_add(int, int);` to declare the function.
    In `C++`, we use `extern "C" int shared_add(int, int);` to declare the function.

the build script is in the `examples/peek` dir, you can try it by yourself.



