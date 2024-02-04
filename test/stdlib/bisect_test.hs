import bisect
import sys

li = [1, 3, 4, 4, 4, 6, 7]
lst = [10, 20, 30, 40, 50]

str = ["a", "b", "b", "c", "d"]


@test
def bisect_left():
    assert bisect.bisect_left(li, 4, 0, len(li)) == 2
    assert bisect.bisect_left(li, 4) == 2
    assert bisect.bisect_left(str, "b", 0, len(str)) == 1
    assert bisect.bisect_left(lst, 25, 1, 3) == 2

    # precomputed cases
    assert bisect.bisect_left(List[int](), 1, 0, 0) == 0
    assert bisect.bisect_left([1], 0, 0, 1) == 0
    assert bisect.bisect_left([1], 1, 0, 1) == 0
    assert bisect.bisect_left([1], 2, 0, 1) == 1
    assert bisect.bisect_left([1, 1], 0, 0, 2) == 0
    assert bisect.bisect_left([1, 1], 1, 0, 2) == 0
    assert bisect.bisect_left([1, 1], 2, 0, 2) == 2
    assert bisect.bisect_left([1, 1, 1], 0, 0, 3) == 0
    assert bisect.bisect_left([1, 1, 1], 1, 0, 3) == 0
    assert bisect.bisect_left([1, 1, 1], 2, 0, 3) == 3
    assert bisect.bisect_left([1, 1, 1, 1], 0, 0, 4) == 0
    assert bisect.bisect_left([1, 1, 1, 1], 1, 0, 4) == 0
    assert bisect.bisect_left([1, 1, 1, 1], 2, 0, 4) == 4
    assert bisect.bisect_left([1, 2], 0, 0, 2) == 0
    assert bisect.bisect_left([1, 2], 1, 0, 2) == 0
    assert bisect.bisect_left([1, 2], 1.5, 0, 2) == 1
    assert bisect.bisect_left([1, 2], 2, 0, 2) == 1
    assert bisect.bisect_left([1, 2], 3, 0, 2) == 2
    assert bisect.bisect_left([1, 1, 2, 2], 0, 0, 4) == 0
    assert bisect.bisect_left([1, 1, 2, 2], 1, 0, 4) == 0
    assert bisect.bisect_left([1, 1, 2, 2], 1.5, 0, 4) == 2
    assert bisect.bisect_left([1, 1, 2, 2], 2, 0, 4) == 2
    assert bisect.bisect_left([1, 1, 2, 2], 3, 0, 4) == 4
    assert bisect.bisect_left([1, 2, 3], 0, 0, 3) == 0
    assert bisect.bisect_left([1, 2, 3], 1, 0, 3) == 0
    assert bisect.bisect_left([1, 2, 3], 1.5, 0, 3) == 1
    assert bisect.bisect_left([1, 2, 3], 2, 0, 3) == 1
    assert bisect.bisect_left([1, 2, 3], 2.5, 0, 3) == 2
    assert bisect.bisect_left([1, 2, 3], 3, 0, 3) == 2
    assert bisect.bisect_left([1, 2, 3], 4, 0, 3) == 3
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 0, 0, 10) == 0
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1, 0, 10) == 0
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1.5, 0, 10) == 1
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2, 0, 10) == 1
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2.5, 0, 10) == 3
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3, 0, 10) == 3
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3.5, 0, 10) == 6
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 4, 0, 10) == 6
    assert bisect.bisect_left([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 5, 0, 10) == 10


@test
def bisect_right():
    assert bisect.bisect_right(li, 4, 0, len(li)) == 5
    assert bisect.bisect_right(li, 4) == 5
    assert bisect.bisect_right(str, "b", 0, len(str)) == 3
    assert bisect.bisect_right(lst, 25, 1, 3) == 2

    # precomputed casesrightrt bisect.bisect_right(List[int](), 1, 0, 0) == 0
    assert bisect.bisect_right([1], 0, 0, 1) == 0
    assert bisect.bisect_right([1], 1, 0, 1) == 1
    assert bisect.bisect_right([1], 2, 0, 1) == 1
    assert bisect.bisect_right([1, 1], 0, 0, 2) == 0
    assert bisect.bisect_right([1, 1], 1, 0, 2) == 2
    assert bisect.bisect_right([1, 1], 2, 0, 2) == 2
    assert bisect.bisect_right([1, 1, 1], 0, 0, 3) == 0
    assert bisect.bisect_right([1, 1, 1], 1, 0, 3) == 3
    assert bisect.bisect_right([1, 1, 1], 2, 0, 3) == 3
    assert bisect.bisect_right([1, 1, 1, 1], 0, 0, 4) == 0
    assert bisect.bisect_right([1, 1, 1, 1], 1, 0, 4) == 4
    assert bisect.bisect_right([1, 1, 1, 1], 2, 0, 4) == 4
    assert bisect.bisect_right([1, 2], 0, 0, 2) == 0
    assert bisect.bisect_right([1, 2], 1, 0, 2) == 1
    assert bisect.bisect_right([1, 2], 1.5, 0, 2) == 1
    assert bisect.bisect_right([1, 2], 2, 0, 2) == 2
    assert bisect.bisect_right([1, 2], 3, 0, 2) == 2
    assert bisect.bisect_right([1, 1, 2, 2], 0, 0, 4) == 0
    assert bisect.bisect_right([1, 1, 2, 2], 1, 0, 4) == 2
    assert bisect.bisect_right([1, 1, 2, 2], 1.5, 0, 4) == 2
    assert bisect.bisect_right([1, 1, 2, 2], 2, 0, 4) == 4
    assert bisect.bisect_right([1, 1, 2, 2], 3, 0, 4) == 4
    assert bisect.bisect_right([1, 2, 3], 0, 0, 3) == 0
    assert bisect.bisect_right([1, 2, 3], 1, 0, 3) == 1
    assert bisect.bisect_right([1, 2, 3], 1.5, 0, 3) == 1
    assert bisect.bisect_right([1, 2, 3], 2, 0, 3) == 2
    assert bisect.bisect_right([1, 2, 3], 2.5, 0, 3) == 2
    assert bisect.bisect_right([1, 2, 3], 3, 0, 3) == 3
    assert bisect.bisect_right([1, 2, 3], 4, 0, 3) == 3
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 0, 0, 10) == 0
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1, 0, 10) == 1
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1.5, 0, 10) == 1
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2, 0, 10) == 3
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2.5, 0, 10) == 3
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3, 0, 10) == 6
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3.5, 0, 10) == 6
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 4, 0, 10) == 10
    assert bisect.bisect_right([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 5, 0, 10) == 10


@test
def bisect1():
    assert bisect.bisect(li, 4, 0, len(li)) == 5
    assert bisect.bisect(str, "b", 0, len(str)) == 3
    assert bisect.bisect(lst, 25, 1, 3) == 2

    # precomputed casesrightrt bisect.bisect_right(List[int](), 1, 0, 0) == 0
    assert bisect.bisect([1], 0, 0, 1) == 0
    assert bisect.bisect([1], 1, 0, 1) == 1
    assert bisect.bisect([1], 2, 0, 1) == 1
    assert bisect.bisect([1, 1], 0, 0, 2) == 0
    assert bisect.bisect([1, 1], 1, 0, 2) == 2
    assert bisect.bisect([1, 1], 2, 0, 2) == 2
    assert bisect.bisect([1, 1, 1], 0, 0, 3) == 0
    assert bisect.bisect([1, 1, 1], 1, 0, 3) == 3
    assert bisect.bisect([1, 1, 1], 2, 0, 3) == 3
    assert bisect.bisect([1, 1, 1, 1], 0, 0, 4) == 0
    assert bisect.bisect([1, 1, 1, 1], 1, 0, 4) == 4
    assert bisect.bisect([1, 1, 1, 1], 2, 0, 4) == 4
    assert bisect.bisect([1, 2], 0, 0, 2) == 0
    assert bisect.bisect([1, 2], 1, 0, 2) == 1
    assert bisect.bisect([1, 2], 1.5, 0, 2) == 1
    assert bisect.bisect([1, 2], 2, 0, 2) == 2
    assert bisect.bisect([1, 2], 3, 0, 2) == 2
    assert bisect.bisect([1, 1, 2, 2], 0, 0, 4) == 0
    assert bisect.bisect([1, 1, 2, 2], 1, 0, 4) == 2
    assert bisect.bisect([1, 1, 2, 2], 1.5, 0, 4) == 2
    assert bisect.bisect([1, 1, 2, 2], 2, 0, 4) == 4
    assert bisect.bisect([1, 1, 2, 2], 3, 0, 4) == 4
    assert bisect.bisect([1, 2, 3], 0, 0, 3) == 0
    assert bisect.bisect([1, 2, 3], 1, 0, 3) == 1
    assert bisect.bisect([1, 2, 3], 1.5, 0, 3) == 1
    assert bisect.bisect([1, 2, 3], 2, 0, 3) == 2
    assert bisect.bisect([1, 2, 3], 2.5, 0, 3) == 2
    assert bisect.bisect([1, 2, 3], 3, 0, 3) == 3
    assert bisect.bisect([1, 2, 3], 4, 0, 3) == 3
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 0, 0, 10) == 0
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1, 0, 10) == 1
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 1.5, 0, 10) == 1
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2, 0, 10) == 3
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 2.5, 0, 10) == 3
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3, 0, 10) == 6
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 3.5, 0, 10) == 6
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 4, 0, 10) == 10
    assert bisect.bisect([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 5, 0, 10) == 10


@test
def insort_left():
    bisect.insort_left(li, 5)
    assert li == [1, 3, 4, 4, 4, 5, 6, 7]


@test
def insort_right():
    li = [1, 3, 4, 4, 4, 6, 7]
    bisect.insort_right(li, 0, 0, len(li))
    assert li == [0, 1, 3, 4, 4, 4, 6, 7]
    bisect.insort_right(li, 10)
    assert li == [0, 1, 3, 4, 4, 4, 6, 7, 10]


@test
def insort():
    li = [1, 3, 4, 4, 4, 6, 7]
    bisect.insort(li, 0, 0, len(li))
    assert li == [0, 1, 3, 4, 4, 4, 6, 7]
    bisect.insort(li, 10)
    assert li == [0, 1, 3, 4, 4, 4, 6, 7, 10]


bisect_left()
bisect_right()
bisect1()
insort_left()
insort_right()
insort()
