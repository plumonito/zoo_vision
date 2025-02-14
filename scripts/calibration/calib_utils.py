import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass


def imread_rgb(name):
    m = cv2.imread(name)
    return cv2.cvtColor(m, cv2.COLOR_BGR2RGB)


# Homogenous coordinates utilities
def to_h(x, hvalue=1):
    if len(x.shape) == 1:
        return np.concatenate([x, np.full((1,), hvalue)])
    else:
        count = x.shape[0]
        return np.concatenate([x, np.full((count, 1), hvalue)], axis=1)


def from_h(x):
    if len(x.shape) == 1:
        return x[0:-1] / x[-1]
    else:
        return x[:, 0:-1] / x[:, [-1]]


def hmult(A, b, keep_h=False):
    np.testing.assert_equal(len(A.shape), 2)
    np.testing.assert_equal(A.shape[1], b.shape[-1] + 1)
    bh = to_h(b)
    Abh = (A @ bh.T).T
    if not keep_h:
        return from_h(Abh)
    else:
        return Abh


def test_hfuncs():
    print("Homogenous utilities tests")
    a2 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    print(to_h(a2))
    print(from_h(to_h(a2)))
    A = np.eye(3)
    hmult(A, np.ones((2,)))
    hmult(A, np.ones((10, 2)))
