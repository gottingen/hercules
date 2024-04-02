.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _welcome:

Welcome to EA Hercules
=======================

take a glance at the following simple example to get a sense of how to use EA Hercules.
a simple python code:

.. code-block:: python

    from time import time

    def fib(n):
        return n if n < 2 else fib(n - 1) + fib(n - 2)

    t0 = time()
    ans = fib(40)
    t1 = time()
    print(f'Computed fib(40) = {ans} in {t1 - t0} seconds.')


now, we run it by python and hercules:

.. code-block:: bash

    $ python3 fib.py
    Computed fib(40) = 102334155 in 11.951828241348267 seconds.

    $ hercules run -m 1 fib.py
    Computed fib(40) = 102334155 in 0.238125 seconds.
    $ hercules run -m 0 fib.py
    Computed fib(40) = 102334155 in 0.510346 seconds.

the `-m` option is used to specify the running mode, `1` for running release mode and `0` for running debug mode.


