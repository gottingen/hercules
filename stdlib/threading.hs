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

@tuple
class Lock:
    p: cobj

    def __new__() -> Lock:
        return (_C.hs_lock_new(),)

    def acquire(self, block: bool = True, timeout: float = -1.0) -> bool:
        if timeout >= 0.0 and not block:
            raise ValueError("can't specify a timeout for a non-blocking call")
        return _C.hs_lock_acquire(self.p, block, timeout)

    def release(self):
        _C.hs_lock_release(self.p)

    def __enter__(self):
        self.acquire()

    def __exit__(self):
        self.release()

@tuple
class RLock:
    p: cobj

    def __new__() -> RLock:
        return (_C.hs_rlock_new(),)

    def acquire(self, block: bool = True, timeout: float = -1.0) -> bool:
        if timeout >= 0.0 and not block:
            raise ValueError("can't specify a timeout for a non-blocking call")
        return _C.hs_rlock_acquire(self.p, block, timeout)

    def release(self):
        _C.hs_rlock_release(self.p)

    def __enter__(self):
        self.acquire()

    def __exit__(self):
        self.release()

def active_count() -> int:
    from openmp import get_num_threads
    return get_num_threads()

def get_native_id() -> int:
    from openmp import get_thread_num
    return get_thread_num()

def get_ident() -> int:
    return get_native_id() + 1
