

@ccexport
class Foo:
    x: int

    def __init__(self, x: int) -> None:
        self.x = x

    def __str__(self) -> str:
        return f"Foo({self.x})"


foo = Foo(42)
print(foo)

def bar() -> None:
    foo = Foo(42)
    print(foo)
