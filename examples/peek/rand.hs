import random
def random_number():
    return random.randint(0, 20)

def match_ex():
    for _ in range(10):
        match random_number():  # assuming that the type of this expression is int
            case 1:         # is it 1?
                print('hi')
            case 2 ... 10:  # is it 2, 3, 4, 5, 6, 7, 8, 9 or 10?
                print('wow!')
            case _:         # "default" case
                print('meh...')


match_ex()