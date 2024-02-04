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

from internal.types.optional import unwrap

# Infinite iterators

@inline
def count(start: T = 0, step: T = 1, T: type) -> Generator[T]:
    """
    Return a count object whose ``__next__`` method returns consecutive values.
    """
    n = start
    while True:
        yield n
        n += step

@inline
def cycle(iterable: Generator[T], T: type) -> Generator[T]:
    """
    Cycles repeatedly through an iterable.
    """
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        for element in saved:
            yield element

@inline
def repeat(object: T, times: Optional[int] = None, T: type) -> Generator[T]:
    """
    Make an iterator that returns a given object over and over again.
    """
    if times is None:
        while True:
            yield object
    else:
        for i in range(times):
            yield object

# Iterators terminating on the shortest input sequence

@inline
def accumulate(iterable: Generator[T], func=lambda a, b: a + b, initial=0, T: type):
    """
    Make an iterator that returns accumulated sums, or accumulated results
    of other binary functions (specified via the optional func argument).
    """
    total = initial
    yield total
    for element in iterable:
        total = func(total, element)
        yield total

@inline
@overload
def accumulate(iterable: Generator[T], func=lambda a, b: a + b, T: type):
    """
    Make an iterator that returns accumulated sums, or accumulated results
    of other binary functions (specified via the optional func argument).
    """
    total: Optional[T] = None
    for element in iterable:
        total = element if total is None else func(unwrap(total), element)
        yield unwrap(total)

@tuple
class chain:
    """
    Make an iterator that returns elements from the first iterable until it is exhausted,
    then proceeds to the next iterable, until all of the iterables are exhausted.
    """

    @inline
    def __new__(*iterables):
        for it in iterables:
            for element in it:
                yield element

    @inline
    def from_iterable(iterables):
        for it in iterables:
            for element in it:
                yield element

@inline
def compress(
    data: Generator[T], selectors: Generator[B], T: type, B: type
) -> Generator[T]:
    """
    Return data elements corresponding to true selector elements.
    Forms a shorter iterator from selected data elements using the selectors to
    choose the data elements.
    """
    for d, s in zip(data, selectors):
        if s:
            yield d

@inline
def dropwhile(
    predicate: Callable[[T], bool], iterable: Generator[T], T: type
) -> Generator[T]:
    """
    Drop items from the iterable while predicate(item) is true.
    Afterwards, return every element until the iterable is exhausted.
    """
    b = False
    for x in iterable:
        if not b and not predicate(x):
            b = True
        if b:
            yield x

@inline
def filterfalse(
    predicate: Callable[[T], bool], iterable: Generator[T], T: type
) -> Generator[T]:
    """
    Return those items of iterable for which function(item) is false.
    """
    for x in iterable:
        if not predicate(x):
            yield x

# TODO: fix this once Optional[Callable] lands
@inline
def groupby(iterable, key=Optional[int]()):
    """
    Make an iterator that returns consecutive keys and groups from the iterable.
    """
    currkey = None
    group = []

    for currvalue in iterable:
        k = currvalue if isinstance(key, Optional) else key(currvalue)
        if currkey is None:
            currkey = k
        if k != unwrap(currkey):
            yield unwrap(currkey), group
            currkey = k
            group = []
        group.append(currvalue)
    if currkey is not None:
        yield unwrap(currkey), group

def islice(iterable: Generator[T], stop: Optional[int], T: type) -> Generator[T]:
    """
    Make an iterator that returns selected elements from the iterable.
    """
    if stop is not None and stop.__val__() < 0:
        raise ValueError(
            "Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize."
        )
    i = 0
    for x in iterable:
        if stop is not None and i >= stop.__val__():
            break
        yield x
        i += 1

@overload
def islice(
    iterable: Generator[T],
    start: Optional[int],
    stop: Optional[int],
    step: Optional[int] = None,
    T: type,
) -> Generator[T]:
    """
    Make an iterator that returns selected elements from the iterable.
    """
    from sys import maxsize

    start: int = 0 if start is None else start
    stop: int = maxsize if stop is None else stop
    step: int = 1 if step is None else step
    have_stop = False

    if start < 0 or stop < 0:
        raise ValueError(
            "Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize."
        )
    elif step < 0:
        raise ValueError("Step for islice() must be a positive integer or None.")

    it = range(start, stop, step)
    N = len(it)
    idx = 0
    b = -1

    if N == 0:
        for i, element in zip(range(start), iterable):
            pass
        return

    nexti = it[0]
    for i, element in enumerate(iterable):
        if i == nexti:
            yield element
            idx += 1
            if idx >= N:
                b = i
                break
            nexti = it[idx]

    if b >= 0:
        for i, element in zip(range(b + 1, stop), iterable):
            pass

@inline
def starmap(function, iterable):
    """
    Return an iterator whose values are returned from the function
    evaluated with an argument tuple taken from the given sequence.
    """
    for args in iterable:
        yield function(*args)

@inline
def takewhile(
    predicate: Callable[[T], bool], iterable: Generator[T], T: type
) -> Generator[T]:
    """
    Return successive entries from an iterable as long as the predicate evaluates to true for each entry.
    """
    for x in iterable:
        if predicate(x):
            yield x
        else:
            break

def tee(iterable: Generator[T], n: int = 2, T: type) -> List[Generator[T]]:
    """
    Return n independent iterators from a single iterable.
    """
    from collections import deque

    it = iter(iterable)
    deques = [deque[T]() for i in range(n)]

    def gen(mydeque: deque[T], T: type) -> Generator[T]:
        while True:
            if not mydeque:  # when the local deque is empty
                if it.__done__():
                    return
                it.__resume__()
                if it.__done__():
                    return
                newval = it.next()
                for d in deques:  # load it to all the deques
                    d.append(newval)
            yield mydeque.popleft()

    return [gen(d) for d in deques]

@inline
def zip_longest(*iterables, fillvalue):
    """
    Make an iterator that aggregates elements from each of the iterables.
    If the iterables are of uneven length, missing values are filled-in
    with fillvalue. Iteration continues until the longest iterable is
    exhausted.
    """
    if staticlen(iterables) == 2:
        a = iter(iterables[0])
        b = iter(iterables[1])
        a_done = False
        b_done = False

        while not a.done():
            a_val = a.next()
            b_val = fillvalue
            if not b_done:
                b_done = b.done()
            if not b_done:
                b_val = b.next()
            yield a_val, b_val

        if not b_done:
            while not b.done():
                yield fillvalue, b.next()

        a.destroy()
        b.destroy()
    else:
        iterators = tuple(iter(it) for it in iterables)
        num_active = len(iterators)
        if not num_active:
            return
        while True:
            values = []
            for it in iterators:
                if it.__done__():  # already done
                    values.append(fillvalue)
                elif it.done():  # resume and check
                    num_active -= 1
                    if not num_active:
                        return
                    values.append(fillvalue)
                else:
                    values.append(it.next())
            yield values

@inline
@overload
def zip_longest(*args):
    """
    Make an iterator that aggregates elements from each of the iterables.
    If the iterables are of uneven length, missing values are filled-in
    with fillvalue. Iteration continues until the longest iterable is
    exhausted.
    """

    def get_next(it):
        if it.__done__() or it.done():
            return None
        return it.next()

    iters = tuple(iter(arg) for arg in args)
    while True:
        done_count = 0
        result = tuple(get_next(it) for it in iters)
        all_none = True
        for a in result:
            if a is not None:
                all_none = False
        if all_none:
            return
        yield result
    for it in iters:
        it.destroy()

# Combinatoric iterators

def _as_list(x):
    if isinstance(x, list):
        return x
    else:
        return list(x)

def product(*iterables, repeat: int):
    if repeat < 0:
        raise ValueError("repeat must be non-negative")

    if repeat == 0:
        nargs = 0
    else:
        nargs = len(iterables)

    npools = nargs * repeat
    indices = Ptr[int](npools)

    pools = list(capacity=npools)
    i = 0

    while i < nargs:
        p = _as_list(iterables[i])
        if len(p) == 0:
            return
        pools.append(p)
        indices[i] = 0
        i += 1

    while i < npools:
        pools.append(pools[i - nargs])
        indices[i] = 0
        i += 1

    result = list(capacity=npools)
    for i in range(npools):
        result.append(pools[i][0])

    while True:
        yield result

        result = result.copy()
        i = npools - 1
        while i >= 0:
            pool = pools[i]
            indices[i] += 1

            if indices[i] == len(pool):
                indices[i] = 0
                result[i] = pool[0]
            else:
                result[i] = pool[indices[i]]
                break

            i -= 1

        if i < 0:
            break

@overload
def product(*iterables, repeat: Static[int] = 1):
    if repeat < 0:
        compile_error("repeat must be non-negative")

    # handle some common cases
    if repeat == 0:
        yield ()
    elif repeat == 1 and staticlen(iterables) == 1:
        it0 = iterables[0]
        for a in it0:
            yield (a,)
    elif repeat == 1 and staticlen(iterables) == 2:
        it0 = iterables[0]
        it1 = iterables[1]
        for a in it0:
            for b in it1:
                yield (a, b)
    elif repeat == 1 and staticlen(iterables) == 3:
        it0 = iterables[0]
        it1 = iterables[1]
        it2 = iterables[2]
        for a in it0:
            for b in it1:
                for c in it2:
                    yield (a, b, c)
    else:
        nargs: Static[int] = staticlen(iterables)
        npools: Static[int] = nargs * repeat
        indices_tuple = (0,) * npools
        indices = Ptr[int](__ptr__(indices_tuple).as_byte())
        pools = tuple(_as_list(it) for it in iterables) * repeat

        for i in staticrange(nargs):
            if len(pools[i]) == 0:
                return

        result = tuple(pool[0] for pool in pools)

        while True:
            yield result

            i = npools - 1
            while i >= 0:
                pool = pools[i]
                indices[i] += 1

                if indices[i] == len(pool):
                    indices[i] = 0
                else:
                    break

                i -= 1

            if i < 0:
                break

            result = tuple(pools[i][indices[i]] for i in staticrange(npools))

def combinations(pool, r: int):
    if r < 0:
        raise ValueError("r must be non-negative")

    pool_list = _as_list(pool)
    n = len(pool)

    if r > n:
        return

    pool = pool_list.arr.ptr
    indices = Ptr[int](r)
    result = list(capacity=r)

    for i in range(r):
        indices[i] = i
        result.append(pool[i])

    while True:
        yield result

        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1

        if i < 0:
            break

        indices[i] += 1

        for j in range(i + 1, r):
            indices[j] = indices[j-1] + 1

        result = result.copy()
        while i < r:
            result[i] = pool[indices[i]]
            i += 1

@overload
def combinations(pool, r: Static[int]):
    def empty(T: type) -> T:
        pass

    if r < 0:
        compile_error("r must be non-negative")

    if isinstance(pool, list):
        pool_list = pool
    else:
        pool_list = list(pool)

    n = len(pool)

    if r > n:
        return

    pool = pool_list.arr.ptr
    indices_tuple = (0,) * r
    indices = Ptr[int](__ptr__(indices_tuple).as_byte())
    result_tuple = (empty(pool.T),) * r
    result = Ptr[pool.T](__ptr__(result_tuple).as_byte())

    for i in range(r):
        indices[i] = i
        result[i] = pool[i]

    while True:
        yield result_tuple

        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1

        if i < 0:
            break

        indices[i] += 1

        for j in range(i + 1, r):
            indices[j] = indices[j-1] + 1

        while i < r:
            result[i] = pool[indices[i]]
            i += 1

def combinations_with_replacement(pool, r: int):
    if r < 0:
        raise ValueError("r must be non-negative")

    pool_list = _as_list(pool)
    n = len(pool)

    if n == 0:
        if r == 0:
            yield List[pool_list.T](capacity=0)
        return

    pool = pool_list.arr.ptr
    indices = Ptr[int](r)
    result = list(capacity=r)

    for i in range(r):
        indices[i] = 0
        result.append(pool[0])

    while True:
        yield result

        i = r - 1
        while i >= 0 and indices[i] == n - 1:
            i -= 1

        if i < 0:
            break

        result = result.copy()
        index = indices[i] + 1
        elem = pool[index]

        while i < r:
            indices[i] = index
            result[i] = elem
            i += 1

@overload
def combinations_with_replacement(pool, r: Static[int]):
    def empty(T: type) -> T:
        pass

    if r < 0:
        compile_error("r must be non-negative")

    if r == 0:
        yield ()
        return

    if isinstance(pool, list):
        pool_list = pool
    else:
        pool_list = list(pool)

    n = len(pool)

    if n == 0:
        return

    pool = pool_list.arr.ptr
    indices_tuple = (0,) * r
    indices = Ptr[int](__ptr__(indices_tuple).as_byte())
    result_tuple = (empty(pool.T),) * r
    result = Ptr[pool.T](__ptr__(result_tuple).as_byte())

    for i in range(r):
        result[i] = pool[0]

    while True:
        yield result_tuple

        i = r - 1
        while i >= 0 and indices[i] == n - 1:
            i -= 1

        if i < 0:
            break

        index = indices[i] + 1
        elem = pool[index]

        while i < r:
            indices[i] = index
            result[i] = elem
            i += 1

def _permutations_non_static(pool, r = None):
    pool_list = _as_list(pool)
    n = len(pool)

    if r is None:
        return _permutations_non_static(pool_list, n)
    elif not isinstance(r, int):
        compile_error("Expected int as r")

    if r < 0:
        raise ValueError("r must be non-negative")

    if r > n:
        return

    indices = Ptr[int](n)
    cycles = Ptr[int](r)

    for i in range(n):
        indices[i] = i

    for i in range(r):
        cycles[i] = n - i

    pool = pool_list.arr.ptr
    result = list(capacity=r)

    for i in range(r):
        result.append(pool[i])

    while True:
        yield result

        if n == 0:
            break

        result = result.copy()
        i = r - 1
        while i >= 0:
            cycles[i] -= 1
            if cycles[i] == 0:
                index = indices[i]
                for j in range(i, n - 1):
                    indices[j] = indices[j+1]
                indices[n-1] = index
                cycles[i] = n - i
            else:
                j = cycles[i]
                index = indices[i]
                indices[i] = indices[n - j]
                indices[n - j] = index

                for k in range(i, r):
                    index = indices[k]
                    result[k] = pool[index]

                break
            i -= 1

        if i < 0:
            break

def _permutations_static(pool, r: Static[int]):
    def empty(T: type) -> T:
        pass

    pool_list = _as_list(pool)
    n = len(pool)

    if r < 0:
        raise compile_error("r must be non-negative")

    if r > n:
        return

    indices = Ptr[int](n)
    cycles_tuple = (0,) * r
    cycles = Ptr[int](__ptr__(cycles_tuple).as_byte())

    for i in range(n):
        indices[i] = i

    for i in range(r):
        cycles[i] = n - i

    pool = pool_list.arr.ptr
    result_tuple = (empty(pool.T),) * r
    result = Ptr[pool.T](__ptr__(result_tuple).as_byte())

    for i in range(r):
        result[i] = pool[i]

    while True:
        yield result_tuple

        if n == 0:
            break

        i = r - 1
        while i >= 0:
            cycles[i] -= 1
            if cycles[i] == 0:
                index = indices[i]
                for j in range(i, n - 1):
                    indices[j] = indices[j+1]
                indices[n-1] = index
                cycles[i] = n - i
            else:
                j = cycles[i]
                index = indices[i]
                indices[i] = indices[n - j]
                indices[n - j] = index

                for k in range(i, r):
                    index = indices[k]
                    result[k] = pool[index]

                break
            i -= 1

        if i < 0:
            break

def permutations(pool, r = None):
    if isinstance(pool, Tuple) and r is None:
        return _permutations_static(pool, staticlen(pool))
    else:
        return _permutations_non_static(pool, r)

@overload
def permutations(pool, r: Static[int]):
    return _permutations_static(pool, r)
