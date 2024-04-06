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

# mkdir lib
if [ -d "${BUILD_PATH}" ]; then
  rm -rf "${BUILD_PATH}"
fi

mkdir -p "${BUILD_PATH}"

# build lib
cd "${BUILD_PATH}"
cmake ../ -DBUILD_TEST=OFF
make -j8

# install to deploy

cmake --install . --prefix ${BUILD_PATH}/hs-install
cd ${BUILD_PATH}/hs-install
tar -cjvf hs-deploy.tar.bz2 *
VFILE=${ROOT_PATH}/hercules/config/config.h
VM=`grep "HERCULES_VERSION_MAJOR" hercules/config/config.h |awk '{print $3}'`
VN=`grep "HERCULES_VERSION_MINOR" hercules/config/config.h |awk '{print $3}'`
VP=`grep "HERCULES_VERSION_PATCH" hercules/config/config.h |awk '{print $3}'`
VERSION="${VM}.${VN}.${VP}"
PLATFORM_ORG=$(uname -s)
PLATFORM="${PLATFORM_ORG,,}"
ARCH=$(uname -m)
OUT_NAME="hercules_${PLATFORM}_${ARCH}_${VERSION}.sh"

cat ${THIS_PATH}/installer.sh ${BUILD_PATH}/hs-install/hs-deploy.tar.bz2 > ${BUILD_PATH}/${OUT_NAME}
chmod +x ${BUILD_PATH}/${OUT_NAME}

