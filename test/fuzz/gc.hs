
class Gc:
    va:int
    def __new__(a:int) -> Gc:
        return Gc(a)

    def __del__(self):
        print("Destructor called")


def test_gc():
    a = Gc(10)
    print(a.va)
    del a
    print("End of function")


test_gc()