.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _cmd-line:

Command Line Interface
===========================

The `hercules` command is the main entry point for running Hercules. It is a command line tool
that can be used to run Hercules in various modes. The command line interface is designed to be
easy to use and intuitive. The command line interface is designed to be easy to use and intuitive.

The `hercules` command has the following subcommands:

- `hercules run` - Run Hercules in the specified mode.
- `hercules build` - Build the Hercules to c/c++ library.
- `hercules jit` - Run Hercules in the JIT mode.
- `hercules version` - Print the version of Hercules and dependencies information.

take a glance at the help message of the `hercules` command:

.. code-block:: bash

    $ hercules --help
    hercules
    Usage: Hercules Programming Language [OPTIONS] SUBCOMMAND

    Options:
      -h,--help                   Print this help message and exit

    Subcommands:
      version                     Print the version of Hercules
      run                         Run a program interactively
      build                       Build a program
      doc                         Generate program documentation
      jit                         Run a program using JIT
      jupyter                     Start a Jupyter kernel

version
----------------

..  code-block:: bash

    $ hercules version
    +--------------------------------------------------------------------------------------------------------------+
    |                                     hercules is a AOT compiler for python                                    |
    |                                     https://github.com/gottingen/hercules                                    |
    +--------------------------------------------------------------------------------------------------------------+
    | hercules is an AOT compiler for python that compiles python code to native code which can be called from C++ |
    |                             +--------------+----------------+------------------+                             |
    |                             | AOT compiler | Requires C++17 | Apache 2 License |                             |
    |                             +--------------+----------------+------------------+                             |
    +--------------------------------------------------------------------------------------------------------------+
    |                           Author: Jeff                       Email: lijippy@163.com                          |
    +--------------------------------------------------------------------------------------------------------------+
    |                                         Hercules runtime information                                         |
    | +-----------------------+----------------------------------------+----------------------------------------+  |
    | |  Hercules Information :          Compiler Information          |              extra options             |  |
    | +-----------------------+----------------------------------------+----------------------------------------+  |
    | |     version: 0.2.6    :             compiler: GNUC             |            cxx standard: 17            |  |
    | |  build type: Release  :          compiler version: 9.4         |              cxx abi: true             |  |
    | +-----------------------+----------------------------------------+----------------------------------------+  |
    +--------------------------------------------------------------------------------------------------------------+
    |                                           Καλώς ήρθατε στον Ηρακλή!                                          |
    +--------------------------------------------------------------------------------------------------------------+
    |                                    Acknowledgements Third party libraries                                    |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           |   project  |     version    |                           URL                           |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | backtrace  | master         | https://github.com/ianlancetaylor/libbacktrace          |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | bdwgc      | 8.0.5          | https://github.com/ivmai/bdwgc                          |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | bz2        | 1.0.8          | https://www.sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | llvm       | 17.0.6         | https://github.com/llvm/llvm-project                    |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | clang      | 17.0.6         | https://github.com/llvm/llvm-project                    |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | collie     | 0.2.7          | https://github.com/gottingen/collie                     |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | googletest | release-1.12.1 | https://github.com/google/googletest                    |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | re2        | 2022-06-01     | https://github.com/google/re2                           |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | xz         | 5.2.5          | https://github.com/xz-mirror/xz                         |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |           | zlibng     | 2.0.5          | https://github.com/zlib-ng/zlib-ng                      |          |
    |           +------------+----------------+---------------------------------------------------------+          |
    |                         🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥                         |
    +--------------------------------------------------------------------------------------------------------------+

The first part of the output shows the version of Hercules and some basic information about it, such as the author and
the license. and what the Hercules is.

The second part of the output shows the Hercules runtime information, such as the version of Hercules, the build type,
the compiler information, and the extra options., if you build it your self, the compiler information will be different.

the third part of the output shows the acknowledgements of the third-party libraries that Hercules uses. The table lists
the project name, version, and URL of the third-party libraries.

In the middle of the output, there is a welcome message in Greek, which means "Welcome to Hercules!".

run
----------------

The `hercules run` subcommand is used to run Hercules in the specified mode. The `hercules run` subcommand has
the following options:

.. code-block:: bash

    $ hercules run --h
    Run a program interactively
    Usage: Hercules Programming Language run [OPTIONS] [prog_args...]

    Positionals:
      prog_args TEXT ...          program arguments

    Options:
      -h,--help                   Print this help message and exit
      -m,--mode ENUM:value in {debug->0,release->1} OR {0,1}
                                  optimization mode
      -D,--define TEXT ...        Add static variable definitions. The syntax is <name>=<value>
      -d,--disable-opt TEXT ...   Disable the specified IR optimization
      -p,--plugin TEXT ...        Load specified plugin
      --log TEXT                  Enable given log streams
      -n,--numeric ENUM:value in {c->0,py->1} OR {0,1}
                                  numerical semantics
      -l,--lib TEXT ...           Link the specified library


`-m` or `--mode` option is used to specify the optimization mode. The optimization mode can be either `debug` or `release`.
The default optimization mode is `debug`.

`-D` or `--define` option is used to add static variable definitions. The syntax is `<name>=<value>`. You can add multiple
static variable definitions by specifying the `-D` option multiple times.

`-d` or `--disable-opt` option is used to disable the specified IR optimization. You can disable multiple IR optimizations
by specifying the `-d` option multiple times.

`-p` or `--plugin` option is used to load the specified plugin. You can load multiple plugins by specifying the `-p` option
multiple times.

`--log` option is used to enable the given log streams. You can enable multiple log streams by specifying the `--log` option
multiple times.

`-n` or `--numeric` option is used to specify the numerical semantics. The numerical semantics can be either `c` or `py`.
The default numerical semantics is `c`.

`-l` or `--lib` option is used to link the specified library. You can link multiple libraries by specifying the `-l` option
multiple times.

build
----------------

The `hercules build` subcommand is used to build the Hercules to c/c++ library. The `hercules build` subcommand has
the following options:

.. code-block:: bash

    $ hercules build --help
    Build a program
    Usage: Hercules Programming Language build [OPTIONS] [prog_args...]

    Positionals:
      prog_args TEXT ...          program arguments

    Options:
      -h,--help                   Print this help message and exit
      -m,--mode ENUM:value in {debug->0,release->1} OR {0,1}
                                  optimization mode
      -D,--define TEXT ...        Add static variable definitions. The syntax is <name>=<value>
      -d,--disable-opt TEXT ...   Disable the specified IR optimization
      -p,--plugin TEXT ...        Load specified plugin
      --log TEXT                  Enable given log streams
      -n,--numeric ENUM:value in {c->0,py->1} OR {0,1}
                                  numerical semantics
      -l,--lib TEXT ...           Link the specified library
      -F,--flags TEXT             compiler flags
      -o,--output TEXT            output file
      -k,--kind ENUM:value in {bc->1,detect->6,exe->3,lib->4,llvm->0,obj->2,pyext->5} OR {1,6,3,4,0,2,5}
                                  output type
      -y,--py_module TEXT         Python extension module name
      -r,--relocation ENUM:value in {dpic->2,pic->1,ropi->3,ropi-rwpi->5,rwpi->4,static->0} OR {2,1,3,5,4,0}
                                  relocation model

The same options as the `hercules run` subcommand are available in the `hercules build` subcommand. In addition, the
`hercules build` subcommand has the following options:

`-F` or `--flags` option is used to specify the compiler flags. You can specify multiple compiler flags by specifying the
`-F` option multiple times.

`-o` or `--output` option is used to specify the output file.

`-k` or `--kind` option is used to specify the output type. The output type can be either `bc`, `detect`, `exe`, `lib`,
`llvm`, `obj`, or `pyext`. The default output type is `exe`.

`-y` or `--py_module` option is used to specify the Python extension module name.

`-r` or `--relocation` option is used to specify the relocation model. The relocation model can be either `dpic`, `pic`,
`ropi`, `ropi-rwpi`, `rwpi`, or `static`. The default relocation model is `static`.
this option actually is not used in hercules, but pass it to llvm, I will explain it more in the future. just remember that,
if you want to build a shared library, you should use the `pic` relocation model.

jit
----------------

The `hercules jit` subcommand is used to run Hercules in the JIT mode. The `hercules jit` subcommand has the following options:

.. code-block:: bash

    $ hercules jit --help
    Run a program using JIT
    Usage: Hercules Programming Language jit [OPTIONS] [prog_args...]

    Positionals:
      prog_args TEXT ...          program arguments

    Options:
      -h,--help                   Print this help message and exit
      -m,--mode ENUM:value in {debug->0,release->1} OR {0,1}
                                  optimization mode
      -D,--define TEXT ...        Add static variable definitions. The syntax is <name>=<value>
      -d,--disable-opt TEXT ...   Disable the specified IR optimization
      -p,--plugin TEXT ...        Load specified plugin
      --log TEXT                  Enable given log streams
      -n,--numeric ENUM:value in {c->0,py->1} OR {0,1}
                                  numerical semantics
      -l,--lib TEXT ...           Link the specified library


The same options as the `hercules run` subcommand and the `hercules build` subcommand are
available in the `hercules jit` subcommand.


