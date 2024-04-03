
from add import simple_add

print(simple_add(3,2))

#LIB="./libadd_shared.so"
from C import shared_add(int, int)->int
print(shared_add(3,2))