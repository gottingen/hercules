import hercules
from time import time

def is_prime_python(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

@hercules.jit
def is_prime_hercules(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

t0 = time()
ans = sum(1 for i in range(100000, 200000) if is_prime_python(i))
t1 = time()
print(f'[python] {ans} | took {t1 - t0} seconds')

t0 = time()
ans = sum(1 for i in range(100000, 200000) if is_prime_hercules(i))
t1 = time()
print(f'[hercules]  {ans} | took {t1 - t0} seconds')