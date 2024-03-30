#! /usr/bin/env bash
# Copyright 2023 The titan-search Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

LLVM_GIT="https://github.com/llvm/llvm-project.git"

THIS_PATH=$(
    cd $(dirname "$0")
    pwd
)

BUILDD_DIR=${THIS_PATH}/../external
INSTALL_ROOT=${THIS_PATH}/../external

cd ${BUILDD_DIR}

# install llvm
if [ ! -d llvm-project ]; then
    git clone --depth 1 -b llvmorg-17.0.0 ${LLVM_GIT} llvm-project
fi

cmake -S llvm-project/llvm -B llvm-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_TARGETS_TO_BUILD=all

cmake --build llvm-build
cmake --install llvm-build --prefix=${INSTALL_ROOT}/llvm17

# install clang

cmake -S llvm-project/clang -B clang-build \
    -G Ninja     -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLIBCLANG_BUILD_STATIC=ON \
     -DLLVM_ENABLE_PIC=OFF

cmake --build clang-build
cmake --install clang-build --prefix=${INSTALL_ROOT}/clang17

