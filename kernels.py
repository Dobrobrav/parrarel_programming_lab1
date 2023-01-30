from typing import Final

import numpy as np

RELIEF_KERNEL: Final = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
], dtype='int16')

BLUR_KERNEL: Final = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])

SHOW_EDGES_KERNEL: Final = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
])

INCREASE_CONTRAST_KERNEL: Final = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
])
