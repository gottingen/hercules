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

set -xue
set -o pipefail

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/../
BUILD_PATH=${ROOT_PATH}/build

cd ${BUILD_PATH}

# xeus
XEUS_VERSION=3.0.5
git clone https://github.com/jupyter-xeus/xeus.git
cd xeus
git checkout ${XEUS_VERSION}
git apply --reject --whitespace=fix  ${ROOT_PATH}/patch/xeus.patch
cd ${BUILD_PATH}
cp -rf xeus ${ROOT_PATH}/3rd/xeus-${XEUS_VERSION}

# xeus-zmq

XEUS_ZMQ_VERSION=1.0.3
git clone https://github.com/jupyter-xeus/xeus-zmq.git
cd xeus-zmq
git checkout ${XEUS_ZMQ_VERSION}
git apply --reject --whitespace=fix  ${ROOT_PATH}/patch/xeus-zmq.patch
cd ${BUILD_PATH}
cp -rf xeus-zmq ${ROOT_PATH}/3rd/xeus-zmq-${XEUS_ZMQ_VERSION}

