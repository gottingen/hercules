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

def _heapify(
    arr: Array[T], begin: int, end: int, keyf: Callable[[T], S], T: type, S: type
):
    """
    Makes the array a heap from [begin, end).
    """
    root = begin
    left = 2 * begin + 1
    right = 2 * begin + 2

    if left < end and keyf(arr[root]) < keyf(arr[left]):
        root = left

    if right < end and keyf(arr[root]) < keyf(arr[right]):
        root = right

    if root != begin:
        arr[begin], arr[root] = arr[root], arr[begin]
        _heapify(arr, root, end, keyf)

def _heap_sort(
    arr: Array[T], begin: int, end: int, keyf: Callable[[T], S], T: type, S: type
):
    if end - begin < 2:
        return

    arr = arr.slice(begin, end)
    end -= begin
    begin = 0

    i = end // 2 - 1
    while i >= 0:
        _heapify(arr, i, end, keyf)
        i -= 1

    i = end - 1
    while i >= 0:
        arr[i], arr[0] = arr[0], arr[i]
        _heapify(arr, 0, i, keyf)
        i -= 1

def heap_sort_array(
    collection: Array[T], size: int, keyf: Callable[[T], S], T: type, S: type
):
    """
    Heap Sort
    Sorts the array inplace.
    """
    _heap_sort(collection, 0, size, keyf)

def heap_sort_inplace(
    collection: List[T], keyf: Callable[[T], S], T: type, S: type
):
    """
    Heap Sort
    Sorts the list inplace.
    """
    heap_sort_array(collection.arr, collection.len, keyf)

def heap_sort(collection: List[T], keyf: Callable[[T], S], T: type, S: type) -> List[T]:
    """
    Heap Sort
    Returns a sorted list.
    """
    newlst = collection.__copy__()
    heap_sort_inplace(newlst, keyf)
    return newlst
