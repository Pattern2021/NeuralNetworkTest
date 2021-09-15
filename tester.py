import numpy as np

def test1(num):
    try:
        if isinstance(num, np.ndarray):
            pass
        else:
            raise TypeError("Not array")
        if num[0] == 1:
            pass
        else:
            raise ValueError("Must be 1")
        
    except Exception as e:
        print(e)

arr = np.array([2])

test1(arr)
test1(1)