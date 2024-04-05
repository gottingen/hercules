
class EnumTest:
    evalue: int

    OK: ClassVar[int] = 0
    ERROR: ClassVar[int] = 1

    def __init__(self) -> None:
        self.evalue = self.OK

    def __init__(self, ev: int) -> None:
        self.evalue = ev

    def __eq__(self, other: int) -> bool:
        return self.evalue == other

    def __eq__(self, other: EnumTest) -> bool:
        return self.evalue == other.evalue

    def __ne__(self, other: int) -> bool:
        return self.evalue != other

    def __ne__(self, other: EnumTest) -> bool:
        return self.evalue != other.evalue

    def __str__(self) -> str:
        return 'EnumTest: ' + str(self.evalue)

    def get_evalue(self) -> int:
        return self.evalue

    def set_evalue(self, v: int) -> None:
        self.evalue = v



def test() -> None:
    e = EnumTest()
    assert e == EnumTest.OK
    e1 = EnumTest()
    print("e: ", e, "e1: ", e1)
    assert e1 == EnumTest.OK
    assert e1 == e
    e1.set_evalue(EnumTest.ERROR)
    assert e1 != e
    assert e1 == EnumTest.ERROR
    assert e1 == 1
    print(e1)
    print(e)


test()
