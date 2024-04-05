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

if [ -d deploy ]; then
  rm -rf deploy
fi
mkdir deploy
lines=31
tail -n+$lines $0 > hs-deploy.tar.bz2
tar xjvf hs-deploy.tar.bz2 -C deploy

if [ ! -d ${HOME}/.hercules ]; then
  mkdir -p ${HOME}/.hercules
fi

cp -r deploy/* ${HOME}/.hercules
rm -rf deploy hs-deploy.tar.bz2
exit 0
