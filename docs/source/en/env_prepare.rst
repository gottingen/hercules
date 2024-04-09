.. Copyright 2024 The Elastic AI Search Authors.
.. Licensed under the Apache License, Version 2.0 (the "License");

.. _env_prepare:

ubuntu development environment preparation
==========================================

build essential
---------------

llvm installation
-----------------

ubuntu 20.04 docker build
---------------------------

.. code-block:: bash

    docker run -it ubuntu:20.04
    cd /root
    apt update
    apt install -y build-essential
    apt install wget -y
    apt install unzip zip -y
    apt install -y libssl-dev
    apt install -y libsodium-dev
    apt install -y uuid-dev
    apt install -y git
    apt install software-properties-common
    add-apt-repository ppa:deadsnakes/ppa
    apt install python3-pip -y
    pip3 install --upgrade pip
    apt install -y python3.8
    pip3 install jupyter
    # https://zhuanlan.zhihu.com/p/74243731
    # install cmake
    wget https://github.com/Kitware/CMake/archive/refs/tags/v3.23.4.zip
    unzip v3.23.4.zip
    cd CMake-3.23.4
    ./bootstrap
    make -j 4
    make install
    cd /root
    rm -rf CMake-3.23.4 v3.23.4.zip
    # install llvm
    git clone https://github.com/llvm/llvm-project.git
    cd llvm-project
    git checkout llvmorg-17.0.6
    cd ..
    cmake -S llvm-project/llvm -B llvm-project/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_TARGETS_TO_BUILD=all \
    -DLIBCLANG_BUILD_STATIC=ON
    cmake --build llvm-project/build -j 4
    cmake --install llvm-project/build --prefix /opt/ea


centos 7 docker build
--------------------

.. code-block:: bash

    docker run -it centos:7
    cd /root
    yum update
    yum install -y epel-release wget unzip zip git gcc gcc-c++ make libsodium-devel libuuid-devel
    yum install bzip2-devel libffi-devel zlib-devel sqlite-devel readline-devel -y
    yum install tk-devel gdbm-devel db4-devel libpcap-devel xz-devel -y
    # openssl
    wget https://www.openssl.org/source/openssl-1.1.1.tar.gz
    tar -xvf openssl-1.1.1.tar.gz
    cd openssl-1.1.1
    ./config --prefix=/usr/local/openssl --openssldir=/usr/local/openssl shared zlib
    make -j 4
    make install
    ln -s /usr/local/openssl/bin/openssl /usr/bin/openssl
    ln -s /usr/local/openssl/include/openssl /usr/include/openssl
    ln -s /usr/local/openssl/lib/libssl.so /usr/lib64/libssl.so
    ln -s /usr/local/openssl/lib/libcrypto.so /usr/lib64/libcrypto.so
    cd /root
    curl -O https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
    tar -xvf Python-3.8.12.tgz
    cd Python-3.8.12
    ./configure --enable-shared #--enable-optimizations
    make -j 4
    make altinstall
    ln -s /usr/local/bin/python3.8 /usr/bin/python3
    ln -s /usr/local/bin/pip3.8 /usr/bin/pip3
    pip3 install --upgrade pip
    pip3 install jupyter
    # https://zhuanlan.zhihu.com/p/74243731
    # install cmake
    wget


.. code-block:: bash

    cmake -S llvm-project/llvm -B llvm-project/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="all" \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_TARGETS_TO_BUILD=all \
    -DLIBCLANG_BUILD_STATIC=ON

    cmake --build llvm-project/build
