
@export
def test_np(a : int):
    from python import numpy as np
    b = np.array([1, 3])
    print(str(b + a))
    print(str(b - a))
