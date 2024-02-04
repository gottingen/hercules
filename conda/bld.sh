#!/bin/bash
#
# Copyright 2023 The Carbin Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -e

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DCMAKE_BUILD_TYPE=Release \
        -DCARBIN_BUILD_TEST=OFF \
        -DCARBIN_BUILD_BENCHMARK=OFF \
        -DCARBIN_BUILD_EXAMPLES=OFF \
        -DCARBIN_USE_CXX11_ABI=ON \
        -DBUILD_SHARED_LIBRARY=ON \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_STATIC_LIBRARY=OFF

cmake --build .
cmake --build . --target install