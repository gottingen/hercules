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

def _insertion_sort(
    arr: Array[T], begin: int, end: int, keyf: Callable[[T], S], T: type, S: type
):
    i = begin + 1
    while i < end:
        x = arr[i]
        j = i - 1
        while j >= begin and keyf(x) < keyf(arr[j]):
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = x
        i += 1

def insertion_sort_array(
    collection: Array[T], size: int, keyf: Callable[[T], S], T: type, S: type
):
    """
    Insertion Sort
    Sorts the array inplace.
    """
    _insertion_sort(collection, 0, size, keyf)

def insertion_sort_inplace(
    collection: List[T], keyf: Callable[[T], S], T: type, S: type
):
    """
    Insertion Sort
    Sorts the list inplace.
    """
    insertion_sort_array(collection.arr, collection.len, keyf)

def insertion_sort(
    collection: List[T], keyf: Callable[[T], S], T: type, S: type
) -> List[T]:
    """
    Insertion Sort
    Returns the sorted list.
    """
    newlst = collection.__copy__()
    insertion_sort_inplace(newlst, keyf)
    return newlst
