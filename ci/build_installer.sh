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
OUTPUT_PATH=${ROOT_PATH}/output
INSTALL_PATH=${ROOT_PATH}/output/hercules
BUILD_PATH=${ROOT_PATH}/build
JUPYTER_PATH=${ROOT_PATH}/jupyter
JUPYTER_BUILD_PATH=${ROOT_PATH}/jupyter/build

# mkdir lib
if [ -d "${BUILD_PATH}" ]; then
  rm -rf "${BUILD_PATH}"
fi
if [ -d "${OUTPUT_PATH}" ]; then
  rm -rf "${OUTPUT_PATH}"
fi

if [ -d "${JUPYTER_BUILD_PATH}" ]; then
  rm -rf "${JUPYTER_BUILD_PATH}"
fi

mkdir -p "${BUILD_PATH}"
mkdir -p "${JUPYTER_BUILD_PATH}"
# build lib
cmake -S ${ROOT_PATH} -B ${BUILD_PATH} -DBUILD_TEST=OFF -DCMAKE_PREFIX_PATH=/opt/EA
cmake --build ${BUILD_PATH} -j 4
cmake --install  ${BUILD_PATH} --prefix ${INSTALL_PATH}


VFILE=${ROOT_PATH}/hercules/config/config.h
VM=`grep "HERCULES_VERSION_MAJOR" ${VFILE} |awk '{print $3}'`
VN=`grep "HERCULES_VERSION_MINOR" ${VFILE} |awk '{print $3}'`
VP=`grep "HERCULES_VERSION_PATCH" ${VFILE} |awk '{print $3}'`
VERSION="${VM}.${VN}.${VP}"
PLATFORM_ORG=$(uname -s)
PLATFORM="${PLATFORM_ORG,,}"
ARCH=$(uname -m)
OUT_NAME="hercules_${PLATFORM}_${ARCH}_${VERSION}.sh"

cd ${INSTALL_PATH}
tar -cjvf ${OUTPUT_PATH}/hs-deploy.tar.bz2 *
cat ${THIS_PATH}/installer.sh ${OUTPUT_PATH}/hs-deploy.tar.bz2 > ${OUTPUT_PATH}/${OUT_NAME}
chmod +x ${OUTPUT_PATH}/${OUT_NAME}
echo "Installer is at ${OUTPUT_PATH}/${OUT_NAME}"

