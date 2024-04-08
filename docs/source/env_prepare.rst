.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _env_prepare:

ubuntu development environment preparation
==========================================

build essential
---------------

llvm installation
-----------------

build llvm from source

.. code-block:: bash

    cmake -S llvm-project/llvm -B \
        llvm-project/build -G Ninja     \
        -DCMAKE_BUILD_TYPE=Release   \
        -DLLVM_INCLUDE_TESTS=OFF     \
        -DLLVM_ENABLE_RTTI=ON     \
        -DLLVM_ENABLE_ZLIB=OFF     \
        -DLLVM_ENABLE_TERMINFO=OFF     \
        -DLLVM_TARGETS_TO_BUILD=all

    cmake --build llvm-project/build
