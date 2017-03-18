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
    return zip(x_iter, y_iter)


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
            for x, y in iter_line(angle, delta, img_size):
                sinogram[i_angle, i_detector] += img[x, y]
    return sinogram


def reverse_radon(sinogram, width, img_size):
    n_angles = sinogram.shape[0]
    n_detectors = sinogram.shape[1]
    img = np.zeros(shape=(n_angles, img_size, img_size), dtype=np.int64)
    width_px = img_size * width
    for i_angle in range(n_angles):
        if i_angle != 0:
            img[i_angle] = img[i_angle-1].copy()
        for i_detector in range(n_detectors):
            angle = i_angle/n_angles * np.pi
            delta = width_px * (-0.5 + i_detector/n_detectors)
            for x, y in iter_line(angle, delta, img_size):
                img[i_angle][x, y] += sinogram[i_angle, i_detector]
    return img


def draw_line(img, d, alpha, delta, value):
    for x, y in iter_line2(alpha, delta, d):
        img[x, y] = value


def draw_rays(size: int, n_rays: int, alpha: float) -> np.ndarray:
    img = np.zeros((size, size), dtype=float)
    for i in range(n_rays):
        delta = -100 + 200 * i/(n_rays-1)
        val = (i + 1) / n_rays
        for x, y in iter_line(alpha, delta, min(img.shape)):
            img[x, y] = val
    img[size//2, size//2] = 1
    return img


def test():
    size = 400
    n_rays = 20
    img = draw_rays(size, n_rays, np.pi * 3/8)

    plt.imshow(img)
    # plt.imsave(fname="output.png", arr=img, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    test()
