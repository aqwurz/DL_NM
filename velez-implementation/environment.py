#!/usr/bin/env python3

import numba
import numpy as np


class Environment():
    def __init__(self):
        food_ids = [
            [7, 6, 5, 4],  # decision bit 0
            [3, 2, 1, 0],  # decision bit 0
            [6, 4, 2, 0],  # decision bit 2
            [7, 5, 3, 1],  # decision bit 2
            [7, 6, 3, 2],  # decision bit 1
            [5, 4, 1, 0]   # decision bit 1
        ]
        self.foods_summer = np.array(food_ids[np.random.randint(0, 6)])
        self.foods_winter = np.array(food_ids[np.random.randint(0, 6)])
        self.presentation_order = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7]]*30
        )
        [np.random.shuffle(day_order) for day_order in self.presentation_order]


@numba.njit('i8[:](i8)')
def decode_food_id(food_id):
    """Translates an integer representation into a bit representation.

    Args:
        food_id (int): A 3-bit integer.
        switch (bool): If the bits should be inverted, to prevent overfitting
            on the values of the bits.

    Returns:
        np.array: The food as three bits.
    """
    out = -np.ones((3,), dtype=np.int64)
    if food_id >= 4:
        out[0] = 1
        food_id -= 4
    if food_id >= 2:
        out[1] = 1
        food_id -= 2
    if food_id >= 1:
        out[2] = 1
    return out
