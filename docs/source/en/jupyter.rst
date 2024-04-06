.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _jupyter:

jupyter notebook support
=======================

The Hermes library provides support for Jupyter notebooks. This allows you to use the Hercules API in a Jupyter notebook
environment.

The Jupiter notebook support is by its kernel file, which we have install into the $HOME/.hercules/share directory. This
kernel file is used to create a new kernel in the Jupyter notebook environment.

Take a glance at the kernel file with a json format `kernel.json` file to the
directory `$HERCULES_HOME/share/jupyter/kernels/hercules/`:

.. code-block:: json

    {
        "display_name": "Hercules",
        "argv": [
            "hercules",
            "jupyter",
            "{connection_file}"
        ],
        "language": "python"
    }

there are also two other `png` format files in the same directory just for the logo of the kernel, that can be ignored.

To install the kernel file, you can use the following command:

.. code-block:: bash

    cp -r $HERCULES_HOME/share/jupyter/kernels/hercules ${ENV}/share/jupyter/kernels/

where `${ENV}` is the directory where the Jupyter notebook is installed. for example:

* if you are using the Jupyter notebook installed by the `conda` package manager, the `${ENV}` is ${CONDA_PREFIX}.
* if you are using the Jupyter notebook installed by the `pip` package manager, the `${ENV}` may be `/usr/lib/`
* if you are using the Jupyter notebook installed by the `pip` package manager by mark `--user` option, the `${ENV}` may be `~/.local/lib/`

plugins support
----------------

Some time, you have developed some plugins for the Hercules, and you want to use them in the Jupyter notebook. You can
install the plugins in the Jupyter notebook environment by  modifying the `kernel.json` file. and reinstall the kernel
file.

like the following `kernel.json` example:

.. code-block:: json

    {
        "display_name": "Hercules",
        "argv": [
            "hercules",
            "jupyter",
            "{connection_file}",
            "--plugin",
            "plugin1 plugin2 plugin3"
        ],
        "language": "python"
    }