

@__notuple__
class EnumTest:
    value: int

    OK: ClassVar[int] = 0
    ERROR: ClassVar[int] = 1
    def __init__(self) -> None:
        self.value = 0

    def get_value(self) -> int:
        return self.value

    def set_value(self, v: int) -> None:
        self.value = v

    def __eq__(self, other: int) -> bool:
        return self.value == other

    def __eq__(self, other: EnumTest) -> bool:
        return self.value == other.value

    def __ne__(self, other: int) -> bool:
        return self.value != other

    def __ne__(self, other: EnumTest) -> bool:
        return self.value != other.value


def test() -> None:
    e = EnumTest()
    assert e == EnumTest.OK
    e1 = EnumTest()
    e1.set_value(EnumTest.ERROR)
    assert e1 != e
    assert e1 == EnumTest.ERROR
    assert e1.get_value() == 1
    e1.OK = 2
    assert e1.OK == 2

test()
