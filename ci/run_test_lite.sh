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

#set -x
set -ue
set -o pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 [core|parse|transform|stdlib|numerics|hir|all]"
    exit 1
fi

THIS_PATH=$(cd $(dirname "$0"); pwd)
ROOT_PATH=${THIS_PATH}/../
BUILD_PATH=${ROOT_PATH}/build

cd "${ROOT_PATH}"
# export env
PYTHONLIB=`find_libpython`
export HERCULES_PYTHON=${PYTHONLIB}
export PYTHONPATH=${ROOT_PATH}/test/python
echo "HERCULES_PYTHON=${HERCULES_PYTHON}"
echo "PYTHONPATH=${PYTHONPATH}"

case $1 in
  core)
    cp ${BUILD_PATH}/test/hercules_core_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_core_test
    ;;
  parse)
    cp ${BUILD_PATH}/test/hercules_parse_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_parse_test
    ;;
  transform)
    cp ${BUILD_PATH}/test/hercules_transform_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_transform_test
    ;;
  stdlib)
    cp ${BUILD_PATH}/test/hercules_stdlib_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_stdlib_test
    ;;
  numerics)
    cp ${BUILD_PATH}/test/hercules_numerics_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_numerics_test
    ;;
  hir)
    cp ${BUILD_PATH}/test/hir_test ${BUILD_PATH}/
    ${BUILD_PATH}/hir_test
    ;;
  all)
    cp ${BUILD_PATH}/test/hercules_core_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_core_test
    cp ${BUILD_PATH}/test/hercules_parse_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_parse_test
    cp ${BUILD_PATH}/test/hercules_transform_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_transform_test
    cp ${BUILD_PATH}/test/hercules_stdlib_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_stdlib_test
    cp ${BUILD_PATH}/test/hercules_numerics_test ${BUILD_PATH}/
    ${BUILD_PATH}/hercules_numerics_test
    cp ${BUILD_PATH}/test/hir_test ${BUILD_PATH}/
    ${BUILD_PATH}/hir_test
    ;;
  *)
    echo "Usage: $0 [core|parse|transform|stdlib|all]"
    exit 1
    ;;
esac

echo "run test done"




