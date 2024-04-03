.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _installing:

binary install
===============================

update later

source install
===============================

build and install from source, almost third-party dependencies are ingested in the source code
3rd-party directory, unfortunately, some `big` dependencies are not included in the source code,
you need to install them manually.

* llvm
* libclang
* cmake
* g++ >= 9.4
* python3

we have already provided a script to install llvm and libclang,
you can run the following command to install them. see `hercules/ci/build_deps.sh`
you can run it follow the steps below:

make sure you have installed `gcc/g++` and `cmake` and `python3` in your system.

.. code-block:: bash

    git clone https://github.com/gottingen/hercules.git
    cd hercules

and then run the following command:

.. code-block:: bash

        bash hercules/ci/build_deps.sh

this will take a bit of time to install llvm and libclang, you can have a cup of coffee.

after that, you can build and install hercules from source code:

.. code-block:: bash

        mkdir build
        cd build
        cmake ..
        make -j4
        make install

this will install hercules to your system, you can use it in your project.

you can also run the following command to build a release version:

.. code-block:: bash

    ./ci/build_installer.sh

this will generate a `hs-linux.sh` file in the `build` directory, you can
run it to install hercules to your system, Also, you can distribute it to
your friends or you children to install hercules on linux, so they would known that a person named Jeff
have created a wonderful project called Hercules.

after that, don't forget to export PATH to your system, by default, hercules will be installed in `~/.hercules`,
and export the path to your system:

.. code-block:: bash

    export PATH=$PATH:~/.hercules/bin


Now, we have installed hercules to our system, let's move on to the next section. try the most simple example to see if it works.
go to the hercules directory, and run the following command:

.. code-block:: bash

    hercules run examples/fib.hs

also see :ref:`examples <welcome>`
