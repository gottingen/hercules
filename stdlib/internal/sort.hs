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

from algorithms.pdqsort import pdq_sort_inplace
from algorithms.insertionsort import insertion_sort_inplace
from algorithms.heapsort import heap_sort_inplace
from algorithms.qsort import qsort_inplace
from algorithms.timsort import tim_sort_inplace

def sorted(
    v: Generator[T],
    key=Optional[int](),
    reverse: bool = False,
    algorithm: Static[str] = "auto",
    T: type,
) -> List[T]:
    """
    Return a sorted list of the elements in v
    """
    newlist = [a for a in v]
    if not isinstance(key, Optional):
        newlist.sort(key, reverse, algorithm)
    else:
        newlist.sort(reverse=reverse, algorithm=algorithm)
    return newlist

def _is_pdq_compatible(x):
    if (isinstance(x, int) or
        isinstance(x, float) or
        isinstance(x, bool) or
        isinstance(x, byte) or
        isinstance(x, str) or
        isinstance(x, Int) or
        isinstance(x, UInt)):
        return True
    elif isinstance(x, Tuple):
        for a in x:
            if not _is_pdq_compatible(a):
                return False
        return True
    else:
        return False

def _sort_list(
    self: List[T], key: Callable[[T], S], algorithm: Static[str], T: type, S: type
):
    if algorithm == "tim" or algorithm == "auto":
        tim_sort_inplace(self, key)
    elif algorithm == "pdq":
        pdq_sort_inplace(self, key)
    elif algorithm == "insertion":
        insertion_sort_inplace(self, key)
    elif algorithm == "heap":
        heap_sort_inplace(self, key)
    elif algorithm == "quick":
        qsort_inplace(self, key)
    else:
        compile_error("invalid sort algorithm")

@extend
class List:
    def sort(
        self,
        key=Optional[int](),
        reverse: bool = False,
        algorithm: Static[str] = "auto",
    ):
        if isinstance(key, Optional):
            if algorithm == "auto":
                # Python uses Timsort in all cases, but if we
                # know stability does not matter (i.e. sorting
                # primitive type with no key), we will use
                # faster PDQ instead. PDQ is ~50% faster than
                # Timsort for sorting 1B 64-bit ints.
                if self:
                    if _is_pdq_compatible(self[0]):
                        pdq_sort_inplace(self, lambda x: x)
                    else:
                        tim_sort_inplace(self, lambda x: x)
            else:
                _sort_list(self, lambda x: x, algorithm)
        else:
            _sort_list(self, key, algorithm)
        if reverse:
            self.reverse()
