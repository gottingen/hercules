from threading import Lock, RLock

n = 0
f = 0.0
g = f32(0.0)

@atomic
def inc(_):
    global n, f, g
    n += 1
    f += 1.0
    g += f32(1.0)
    return 0

@atomic
def dec(_):
    global n, f, g
    n -= 1
    f -= 1.0
    g -= f32(1.0)
    return 0

lock = Lock()
def inc_lock(_):
    global n
    with lock:
        n += 1
    return 0

def dec_lock(_):
    global n
    with lock:
        n -= 1
    return 0

rlock = RLock()
def inc_rlock(_):
    global n
    with rlock:
        n += 1
    return 0

def dec_rlock(_):
    global n
    with rlock:
        n -= 1
    return 0

def foo(_):
    yield 0

@test
def test_parallel_pipe(m: int):
    global n, f
    n, f = 0, 0.0
    range(m) |> iter ||> inc
    assert n == m
    assert f == float(m)
    assert g == f32(m)
    range(m) |> iter ||> dec
    assert n == 0
    assert f == 0.0
    assert g == f32(0.0)
    range(m) |> iter ||> inc |> dec
    assert n == 0
    assert f == 0.0
    assert g == f32(0.0)

    n = 0
    range(m) |> iter ||> inc_lock
    assert n == m
    range(m) |> iter ||> dec_lock
    assert n == 0
    range(m) |> iter ||> inc_lock |> dec_lock
    assert n == 0

    n = 0
    range(m) |> iter ||> inc_rlock
    assert n == m
    range(m) |> iter ||> dec_rlock
    assert n == 0
    range(m) |> iter ||> inc_rlock |> dec_rlock
    assert n == 0

@test
def test_nested_parallel_pipe(m: int):
    global n
    n = 0
    range(m) |> iter ||> inc |> foo ||> dec
    assert n == 0

test_parallel_pipe(0)
test_parallel_pipe(1)
test_parallel_pipe(10)
test_parallel_pipe(10000)

test_nested_parallel_pipe(0)
test_nested_parallel_pipe(1)
test_nested_parallel_pipe(10)
test_nested_parallel_pipe(10000)
