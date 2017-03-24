#!/usr/bin/python3
import numpy as np
# import math
from typing import Iterable, Tuple
import matplotlib.pyplot as plt


def iter_line(alpha: float, displacement: float, size: int) -> Iterable[Tuple[int, int]]:
    r_circle = size // 2
    max_r = int(np.sqrt((r_circle ** 2) - (displacement ** 2))) - 1
    r_iter = range(-max_r, max_r)
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    dx = -int(displacement * sin_a)
    dy = int(displacement * cos_a)
    x_iter = map(lambda r: r_circle + int(r * cos_a) + dx, r_iter)
    y_iter = map(lambda r: r_circle + int(r * sin_a) + dy, r_iter)
    return zip(x_iter, y_iter), len(r_iter)


def radon_transform(img: np.ndarray, n_angles: int, n_detectors: int, \
                     width: float):
    assert len(img.shape) == 2  # greyscale image
    sinogram = np.zeros(shape=(n_angles, n_detectors), dtype=np.int64)
    img_size = min(img.shape)
    width_px = img_size * width
    for i_angle in range(n_angles):
        for i_detector in range(n_detectors):
            angle = i_angle/n_angles * np.pi
            delta = width_px * (-0.5 + i_detector/n_detectors)
            points, _ = iter_line(angle, delta, img_size)
            for x, y in points:
                sinogram[i_angle, i_detector] += img[x, y]
    return sinogram


# class ReverseRadon:
#     def __init__(self, sinogram, width, img_size):
#         n_angles = sinogram.shape[0]
#         n_detectors = sinogram.shape[1]
#         img = np.zeros(shape=(n_angles, img_size, img_size), dtype=np.int64)
#
#     def step():
#         pass

def reverse_radon(img, sinogram, width, img_size):
    n_angles = sinogram.shape[0]
    n_detectors = sinogram.shape[1]
    width_px = img_size * width
    for i_angle in range(n_angles):
        if i_angle != 0:
            img[i_angle] = img[i_angle-1].copy()
        for i_detector in range(n_detectors):
            angle = i_angle/n_angles * np.pi
            delta = width_px * (-0.5 + i_detector/n_detectors)
            points, npoints = iter_line(angle, delta, img_size)
            for x, y in points:
                img[i_angle][x, y] += sinogram[i_angle, i_detector] / npoints
        yield i_angle


def draw_line(img, size, alpha, delta, value):
    for x, y in iter_line(alpha, delta, size):
        img[x, y] = value


def draw_rays(img_size: int, n_angles: int, n_detectors: int, width: float) -> np.ndarray:
    img = np.zeros((img_size, img_size), dtype=float)

    width_px = img_size * width
    for i_angle in range(n_angles):
        for i_detector in range(n_detectors):
            angle = i_angle/n_angles * np.pi
            delta = width_px * (-0.5 + i_detector/n_detectors)
            value = (i_angle+1)/n_angles
            draw_line(img, img_size, angle, delta, value)
    return img


def test():
    img_size = 400
    n_angles = 20
    n_detectors = 5
    width = 0.4
    img = draw_rays(img_size, n_angles, n_detectors, width)

    plt.imshow(img)
    # plt.imsave(fname="output.png", arr=img, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    test()
