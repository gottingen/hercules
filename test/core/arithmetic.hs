@test
def t1():
    assert 2 + 2 == 4
    assert 3.14 * 2 == 6.28
    assert 2 + 3*2 == 8
    assert 1.0/0 == float('inf')
    assert str(0.0/0) == 'nan'

    assert 5 // 2 == 2
    assert 5 / 2 == 2.5
    assert 5.0 // 2.0 == 2
    assert 5.0 / 2.0 == 2.5
    assert 5 // 2.0 == 2
    assert 5 / 2.0 == 2.5
    assert 5.0 // 2 == 2
    assert 5.0 / 2 == 2.5
    assert int(Int[128](5) // Int[128](2)) == 2
    assert Int[128](5) / Int[128](2) == 2.5
t1()

@test
def test_popcnt():
    assert (42).popcnt() == 3
    assert (123).popcnt() == 6
    assert (0).popcnt() == 0
    assert int.popcnt(-1) == 64
    assert u8(-1).popcnt() == 8
    assert (UInt[1024](0xfffffffffffffff3) * UInt[1024](0xfffffffffffffff3)).popcnt() == 4
    assert UInt[128](-1).popcnt() == 128
test_popcnt()

@test
def test_conversions():
    # int -> int, float, bool, str
    assert int(-42) == -42
    assert float(-42) == -42.0
    assert bool(0) == False
    assert bool(-1) == bool(1) == True
    assert str(-42) == '-42'

    # float -> int, float, bool, str
    assert int(-4.2) == -4
    assert int(4.2) == 4
    assert float(-4.2) == -4.2
    assert bool(0.0) == False
    assert bool(-0.1) == bool(0.1) == True
    assert str(-4.2) == '-4.2'

    # bool -> int, float, bool, str
    assert int(False) == 0
    assert int(True) == 1
    assert float(False) == 0.0
    assert float(True) == 1.0
    assert bool(False) == False
    assert bool(True) == True
    assert str(False) == 'False'
    assert str(True) == 'True'

    # byte -> int, float, bool, str
    assert int(byte(42)) == 42
    assert float(byte(42)) == 42.0
    assert bool(byte(0)) == False
    assert bool(byte(42)) == True
    assert str(byte(42)) == '*'

    # intN -> int, float, bool, str | N < 64
    assert int(i32(-42)) == -42
    assert float(i32(-42)) == -42.0
    assert bool(i32(0)) == False
    assert bool(i32(-1)) == bool(i32(1)) == True
    assert str(i32(-42)) == '-42'

    # intN -> int, float, bool, str | N == 64
    assert int(Int[64](-42)) == -42
    assert float(Int[64](-42)) == -42.0
    assert bool(Int[64](0)) == False
    assert bool(Int[64](-1)) == bool(Int[64](1)) == True
    assert str(Int[64](-42)) == '-42'

    # intN -> int, float, bool, str | N > 64
    assert int(Int[80](-42)) == -42
    assert float(Int[80](-42)) == -42.0
    assert bool(Int[80](0)) == False
    assert bool(Int[80](-1)) == bool(Int[80](1)) == True
    assert str(Int[80](-42)) == '-42'

    # uintN -> int, float, bool, str | N < 64
    assert int(u32(42)) == 42
    assert float(u32(42)) == 42.0
    assert bool(u32(0)) == False
    assert bool(u32(42)) == True
    assert str(u32(42)) == '42'

    # uintN -> int, float, bool, str | N == 64
    assert int(UInt[64](42)) == 42
    assert float(UInt[64](42)) == 42.0
    assert bool(UInt[64](0)) == False
    assert bool(UInt[64](42)) == True
    assert str(UInt[64](42)) == '42'

    # uintN -> int, float, bool, str | N > 64
    assert int(UInt[80](42)) == 42
    assert float(UInt[80](42)) == 42.0
    assert bool(UInt[80](0)) == False
    assert bool(Int[80](42)) == True
    assert str(Int[80](42)) == '42'
test_conversions()

@test
def test_int_pow():
    @nonpure
    def f(n):
        return n

    assert f(3) ** f(2) == 9
    assert f(27) ** f(7) == 10460353203
    assert f(-27) ** f(7) == -10460353203
    assert f(-27) ** f(6) == 387420489
    assert f(1) ** f(0) == 1
    assert f(1) ** f(1000) == 1
    assert f(0) ** f(3) == 0
    assert f(0) ** f(0) == 1

    T1 = Int[512]
    assert f(T1(3)) ** f(T1(2)) == T1(9)
    assert f(T1(27)) ** f(T1(7)) == T1(10460353203)
    assert f(T1(-27)) ** f(T1(7)) == T1(-10460353203)
    assert f(T1(-27)) ** f(T1(6)) == T1(387420489)
    assert f(T1(1)) ** f(T1(0)) == T1(1)
    assert f(T1(1)) ** f(T1(1000)) == T1(1)
    assert f(T1(0)) ** f(T1(3)) == T1(0)
    assert f(T1(0)) ** f(T1(0)) == T1(1)
    assert str(f(T1(31)) ** f(T1(31))) == '17069174130723235958610643029059314756044734431'
    assert str(f(T1(-31)) ** f(T1(31))) == '-17069174130723235958610643029059314756044734431'

    T2 = UInt[200]
    assert f(T2(3)) ** f(T2(2)) == T2(9)
    assert f(T2(27)) ** f(T2(7)) == T2(10460353203)
    assert f(T2(1)) ** f(T2(0)) == T2(1)
    assert f(T2(1)) ** f(T2(1000)) == T2(1)
    assert f(T2(0)) ** f(T2(3)) == T2(0)
    assert f(T2(0)) ** f(T2(0)) == T2(1)
    assert str(f(T2(31)) ** f(T2(31))) == '17069174130723235958610643029059314756044734431'
test_int_pow()

@test
def test_float(F: type):
    x = F(5.5)
    assert str(x) == '5.5'
    assert F(x) == x
    assert F() == F(0.0)
    assert x.__copy__() == x
    assert x.__deepcopy__() == x
    assert int(x) == 5
    assert float(x) == 5.5
    assert bool(x)
    assert not bool(F())
    assert +x == x
    assert -x == F(-5.5)
    assert x + x == F(11.0)
    assert x - F(1.0) == F(4.5)
    assert x * F(3.0) == F(16.5)
    assert x / F(2.0) == F(2.75)
    if F is not float128:  # LLVM ops give wrong results for fp128
        assert x // F(2.0) == F(2.0)
        assert x % F(0.75) == F(0.25)
        assert divmod(x, F(0.75)) == (F(7.0), F(0.25))
    assert x == x
    assert x != F()
    assert x < F(6.5)
    assert x > F(4.5)
    assert x <= F(6.5)
    assert x >= F(4.5)
    assert x >= x
    assert x <= x
    assert abs(x) == x
    assert abs(-x) == x
    assert x.__match__(x)
    assert not x.__match__(F())
    assert hash(x) == hash(5.5)

test_float(float)
test_float(float32)
#test_float(float16)
#test_float(bfloat16)
#test_float(float128)
