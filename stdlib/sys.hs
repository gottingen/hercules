#
# Copyright 2023 EA Authors.
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

argv = list(__argv__, len(__argv__))

stdin = File(_C.seq_stdin())
stdout = File(_C.seq_stdout())
stderr = File(_C.seq_stderr())

def exit(status: int = 0):
    raise SystemExit(status)

maxsize = 0x7FFFFFFFFFFFFFFF
