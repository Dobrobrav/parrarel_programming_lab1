import cv2 as cv
import numpy as np
import kernels
from typing import Final, TypeAlias

NDArray2D: TypeAlias = np.ndarray[np.ndarray[int]]
NDArray3D: TypeAlias = np.ndarray[np.ndarray[np.ndarray[int]]]

SCALE_BY: Final = 2


def main():
    img: NDArray3D = cv.imread('pictures/castle.jpg')

    convolved_img = convolve_img(img, kernel=kernels.RELIEF_KERNEL)
    cv.imwrite('pictures/convolved.jpg', convolved_img)

    scaled_img = scale_img(convolved_img, scale_by=SCALE_BY)

    cv.imwrite('pictures/scaled.jpg', scaled_img)


def convolve_img(img: NDArray3D,
                 kernel: NDArray2D) -> NDArray3D:
    blue, green, red = get_channels(img)

    blue_convolved = convolve_channel(blue, kernel)
    green_convolved = convolve_channel(green, kernel)
    red_convolved = convolve_channel(red, kernel)

    convolved_img = merge_channels(
        blue_convolved, green_convolved, red_convolved
    )

    return convolved_img


def get_channels(img: NDArray3D) -> tuple[NDArray2D, NDArray2D, NDArray2D]:
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    return blue_channel, green_channel, red_channel


def convolve_channel(channel: NDArray2D,
                     kernel: NDArray2D) -> NDArray2D:
    res = multiply_matrices(channel, kernel)
    process_matrix(res)
    return res


def multiply_matrices(a: NDArray2D, b: NDArray2D) -> NDArray2D:
    x_size = len(b)
    y_size = len(b[0])
    x_len = len(a)
    y_len = len(a[0])
    res_matrix_x_len = x_len - x_size + 1
    res_matrix_y_len = y_len - y_size + 1
    new = np.empty((res_matrix_x_len, res_matrix_y_len), dtype='int16')

    for i in range(res_matrix_x_len):
        for j in range(res_matrix_y_len):
            res = multiply_area(a[i:i + x_size, j:j + y_size], b)
            new[i, j] = res

    return new


def process_matrix(matrix: NDArray2D):
    negatives_to_zeros(matrix)


def shift_values_up(matrix: NDArray2D):
    min_value = min(min(row) for row in matrix)

    for row in matrix:
        for i, val in enumerate(row):
            row[i] = val - min_value


def negatives_to_zeros(matrix: NDArray2D):
    for row in matrix:
        for j, val in enumerate(row):
            if val < 0:
                row[j] = 0


def abs_values(matrix: NDArray2D):
    for row in matrix:
        for i, val in enumerate(row):
            row[i] = abs(val)


def normalize(matrix: NDArray2D, ceiling_value: int = 255):
    max_value = max(max(row) for row in matrix)

    for row in matrix:
        for i, val in enumerate(row):
            row[i] = val * (ceiling_value / max_value)


def multiply_area(matrix_1: NDArray2D,
                  matrix_2: NDArray2D) -> NDArray2D:
    x_size = len(matrix_1)
    y_size = len(matrix_2)
    new = np.empty((x_size, y_size), dtype='int16')

    for i in range(x_size):
        for j in range(y_size):
            new[i, j] = matrix_1[i, j] * matrix_2[i, j]

    return sum_matrix_values(new)


def sum_matrix_values(matrix: NDArray2D) -> float:
    return sum(sum(row) for row in matrix)


def merge_channels(b: NDArray2D,
                   g: NDArray2D,
                   r: NDArray2D) -> NDArray3D:
    new = np.empty((len(b), len(b[0]), 3), dtype='int16')

    for i in range(len(new)):
        for j in range(len(new[0])):
            values = (b[i, j], g[i, j], r[i, j])
            new[i, j] = np.array(values, dtype='int16')

    return new


def scale_img(img: NDArray3D, scale_by: int = 2):
    b, g, r = get_channels(img)
    b_scaled = scale_channel(b, scale_by)
    g_scaled = scale_channel(g, scale_by)
    r_scaled = scale_channel(r, scale_by)
    img_scaled = merge_channels(b_scaled, g_scaled, r_scaled)

    return img_scaled


def scale_channel(channel: NDArray2D, scale_by: int):
    channel_x = len(channel)
    channel_y = len(channel[0])
    x = channel_x // scale_by
    y = channel_y // scale_by
    new = np.empty((x, y), dtype='int16')
    for i in range(x):
        for j in range(y):
            new[i, j] = get_avg(
                channel[i * scale_by: (i + 1) * scale_by, j * scale_by: (j + 1) * scale_by]
            )

    return new


def get_avg(matrix: NDArray2D) -> int:
    res = round(sum_matrix_values(matrix) / len(matrix) ** 2)
    return res


if __name__ == '__main__':
    main()
