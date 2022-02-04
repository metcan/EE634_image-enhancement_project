# Import libraries
import logging
import os
import re
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.core.numeric import zeros_like
from itertools import permutations
from numba import njit, gdb_init
import sys
from time import process_time
import math
import pandas

IMSHOW_DIVIDER = 4
ATTENUATION_FACTOR = 9
epsilon = np.finfo(np.double).eps


class metrics:
    def __init__(self) -> None:
        pass

    @staticmethod
    def dicrete_entropy(img) -> float:
        marg = np.histogramdd(np.ravel(img), bins=256)[0] / img.size
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        entropy = -np.sum(np.multiply(marg, np.log2(marg)))
        return entropy

    @staticmethod
    def absolute_mean_brightness_error(input_image, output_image) -> float:
        ambe = np.abs(np.mean(output_image / 255) - np.mean(input_image))
        return ambe

    @staticmethod
    def measurement_of_enhancement(img, block_size) -> float:
        entropy = 0
        how_many = img.shape[0] // block_size
        h, w = img.shape
        assert h % block_size == 0, f"{h} rows is not evenly divisible by {block_size}"
        assert w % block_size == 0, f"{w} cols is not evenly divisible by {block_size}"
        image_blocks = (
            img.reshape(h // block_size, block_size, -1, block_size)
            .swapaxes(1, 2)
            .reshape(-1, block_size, block_size)
        )
        block_min = np.min(image_blocks, axis=(1, 2))
        block_max = np.max(image_blocks, axis=(1, 2))
        b_ratio = np.ones_like(block_max)
        entropy = np.zeros_like(block_max)
        b_ratio = np.where(block_min > 0, block_max / block_min, 1)
        entropy = 20 * np.log(b_ratio)
        return np.sum(entropy) / how_many / how_many

    @staticmethod
    def tenengrad_criterion(img, ksize=3) -> float:
        Gx = cv2.Sobel(
            img,
            ddepth=cv2.CV_16S,
            dx=1,
            dy=0,
            ksize=ksize,
            scale=1,
            delta=0,
            borderType=cv2.BORDER_DEFAULT,
        )
        Gy = cv2.Sobel(
            img,
            ddepth=cv2.CV_16S,
            dx=0,
            dy=1,
            ksize=ksize,
            scale=1,
            delta=0,
            borderType=cv2.BORDER_DEFAULT,
        )
        FM = cv2.addWeighted(Gx, 0.5, Gy, 0.5, 0)
        Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
        Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
        FM = Gx * Gx + Gy * Gy
        mn = np.abs(cv2.mean(FM)[0])
        if np.isnan(mn):
            return np.nanmean(FM)
        return mn


@njit
def get_local_min(image):
    kernel_size = 7
    movements = [
        [row - kernel_size // 2, column - kernel_size // 2]
        for row in range(0, kernel_size, 1)
        for column in range(0, kernel_size, 1)
    ]
    local_min = np.ones_like(image)
    max_y, max_x = local_min.shape
    for iy, ix in np.ndindex(local_min.shape):
        min = 2 ^ 32
        for y, x in movements:
            ny = iy + y
            nx = ix + x
            if max_y > ny >= 0 and max_x > nx >= 0:
                if min > image[ny][nx]:
                    min = image[ny][nx]
        local_min[iy, ix] = min
    return local_min


@njit
def get_local_max(image, kernel_size=7):
    movements = [
        [row - kernel_size // 2, column - kernel_size // 2]
        for row in range(0, kernel_size, 1)
        for column in range(0, kernel_size, 1)
    ]
    local_max = np.zeros_like(image)
    max_y, max_x = local_max.shape
    for iy, ix in np.ndindex(local_max.shape):
        max = 0
        for y, x in movements:
            ny = iy + y
            nx = ix + x
            if max_y > ny >= 0 and max_x > nx >= 0:
                if max < image[ny][nx]:
                    max = image[ny][nx]
        local_max[iy, ix] = max
    return local_max


def load_img(data_path=""):
    return cv2.imread(data_path, 0)


def histogram_equlization(img):
    return cv2.equalizeHist(img)


def normalize_image(img):
    return img / np.max(img)


def merge_images(images):
    res = np.hstack(images)


def display_images(images):
    lenght = len(images)
    x = 2
    y = math.ceil(lenght / 2)
    f, axarr = plt.subplots(y, x)
    for index, image in enumerate(images):
        j = index // 2
        i = index % 2
        if y == 1:
            axarr[index].imshow(image * 255, cmap="gray", vmin=0, vmax=255)
        else:
            axarr[j, i].imshow(image * 255, cmap="gray", vmin=0, vmax=255)
    plt.show()


def show_image(img, window_name="window"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Generate E_i
# Merge E_i to obtain F
def save_image(img, filename):
    cv2.imwrite(filename, img)


def calculate_lambda(attenuation_factor, image, local_min, local_max):
    _local_max = local_max + epsilon
    quotient = 1.0 - (attenuation_factor * local_min * (1.0 / _local_max - 1.0))
    quotient = np.log(quotient)
    divisor = np.log(local_max)
    divisor = np.nan_to_num(divisor) + epsilon
    result = quotient / divisor
    result = np.nan_to_num(result)
    return result


def apply_gaussian_filter(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_laplacian_filter(image, kernel_size=5):
    ddepth = cv2.CV_64FC1  # integer scaling
    return np.abs(cv2.Laplacian(image, ddepth=ddepth, ksize=kernel_size))


def compute_gaussian_pyramid(input_matrix, level=6):
    im = input_matrix.copy()
    gpA = [im]
    for i in range(level):
        im = cv2.pyrDown(im)
        gpA.append(im)
    return gpA


def compute_laplacian_pyramid(input_matrix, level=7):
    gpA = compute_gaussian_pyramid(input_matrix)
    lpA = [gpA[level - 1]]
    for i in range(level - 1, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    return lpA


def compute_exposure_fusion(enhanced_image_matrix_list, weight_matrix_list):
    image_pyramid_list = []
    weight_pyramid_list = []
    result_pyramid = []
    for enhanced_image_matrix, weight_matrix in zip(
        enhanced_image_matrix_list, weight_matrix_list
    ):
        image_pyramid_list.append(
            compute_laplacian_pyramid(enhanced_image_matrix, level=7)
        )
        weight_pyramid_list.append(compute_gaussian_pyramid(weight_matrix, level=6))

    result_list = [None] * len(image_pyramid_list[0])

    for i, weight_pyramid in enumerate(weight_pyramid_list):
        weight_pyramid.reverse()
        image_pyramid = image_pyramid_list[i]
        for j, weight in enumerate(weight_pyramid):
            image = image_pyramid[j]
            result = result_list[j]
            if result is None:
                result = image * weight
            else:
                result += image * weight
            result_list[j] = result
    result = result_list[0]
    for i in range(1, len(result_list)):
        result = cv2.pyrUp(result)
        result = cv2.add(result, result_list[i])
    return result


def calculate_enhanced_image(
    attenuation_factor, image, lambda_array, local_min, local_max
):
    quotient = image - attenuation_factor * local_min
    divisor = np.power(local_max, lambda_array) - attenuation_factor * local_min
    result = quotient / divisor
    result = np.nan_to_num(result)
    return result


def calculate_constrast_level(image, kernel_size=5, epsilon=sys.float_info.epsilon):
    return apply_laplacian_filter(image, kernel_size) / (
        apply_gaussian_filter(image, kernel_size) + epsilon
    )


def calculate_brightness_preservation_metric(image, local_max):
    return np.exp(-(((image - local_max) / np.std(image - local_max)) ** 2))


def normalize_weights(weight_matrix_list):
    weight_sum = np.sum(weight_matrix_list, axis=0)
    zero_sum = np.where(weight_sum == 0)
    zero_sum = np.array(zero_sum).T
    weight_sum_inv = np.ones_like(weight_matrix_list[0]) / (weight_sum + epsilon)
    for i in range(len(weight_matrix_list)):
        weight_matrix_list[i] = weight_sum_inv * weight_matrix_list[i]
        if len(zero_sum) != 0:
            weight_matrix_list[i][zero_sum[:, 0], zero_sum[:, 1]] = 1 / len(
                weight_matrix_list
            )
    return weight_matrix_list


if __name__ == "__main__":
    data_path = "alg_test.png"
    image = load_img(data_path)
    Image_Size = (512, 512)
    image = original = cv2.resize(image, Image_Size, interpolation=cv2.INTER_AREA)
    image = hist_image = histogram_equlization(image)
    image = normalize_image(image)
    local_min = get_local_min(image)
    local_max = get_local_max(image)
    # display_images([local_min, local_max])
    K = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    list_zero_max = np.where(local_max == local_min)
    list_zero_max = np.array(list_zero_max).T
    list_one_max = np.where(local_max == 1)
    list_one_max = np.array(list_one_max).T
    list_min_max = np.where(local_max == local_min)
    list_min_max = np.array(list_min_max).T
    weight_matrix_list = []
    enhanced_image_list = []
    for i in K:
        lambda_array = calculate_lambda(i, image, local_min, local_max)
        if len(list_zero_max) != 0:
            lambda_array[list_zero_max[:, 0], list_zero_max[:, 1]] = 0
        if len(list_one_max) != 0:
            lambda_array[list_zero_max[:, 0], list_zero_max[:, 1]] = 1

        enhanced_image = calculate_enhanced_image(
            i, image, lambda_array, local_min, local_max
        )

        if len(list_min_max) != 0:
            enhanced_image[list_min_max[:, 0], list_min_max[:, 1]] = local_max[
                list_min_max[:, 0], list_min_max[:, 1]
            ]
        enhanced_image_list.append(enhanced_image)
        contrast_level = calculate_constrast_level(enhanced_image)
        brightness_preservation_metric = calculate_brightness_preservation_metric(
            enhanced_image, local_max
        )
        weight_matrix = brightness_preservation_metric * contrast_level
        weight_matrix_list.append(weight_matrix)
        print(
            f"Lambda array max value = {np.max(lambda_array)} and min value = {np.min(lambda_array)}"
        )
        print(
            f"Contrast Level array max value = {np.max(contrast_level)} and min value = {np.min(contrast_level)}"
        )
        print(
            f"Weight array max value = {np.max(weight_matrix)} and min value = {np.min(weight_matrix)}"
        )
    weight_matrix_list = normalize_weights(weight_matrix_list)
    result = compute_exposure_fusion(enhanced_image_list, weight_matrix_list)
    normalizedImg = np.zeros_like(result)
    normalizedImg = np.uint8(
        cv2.normalize(result, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    )
    # show_image(normalizedImg)
    save_image(normalizedImg, "./bite_result.png")
    plt.subplot(131), plt.imshow(original, "gray"), plt.title("Input Image")
    plt.subplot(132), plt.imshow(hist_image, "gray"), plt.title(
        "Histogram Equalization"
    )
    plt.subplot(133), plt.imshow(normalizedImg, "gray"), plt.title("Algorithm Output")
    plt.show()
    plt.savefig("output.jpg")
    # cv2.imwrite("out.png", vis)
    ambe = metrics.absolute_mean_brightness_error(image, normalizedImg)
    de = metrics.dicrete_entropy(normalizedImg)
    eme = metrics.measurement_of_enhancement(normalizedImg, 8)
    ten = metrics.tenengrad_criterion(normalizedImg, ksize=3)
    metrix_results = {"ambe": ambe, "de": de, "eme": eme, "ten": ten}
    print(metrix_results)
    ambe = metrics.absolute_mean_brightness_error(image, hist_image)
    de = metrics.dicrete_entropy(hist_image)
    eme = metrics.measurement_of_enhancement(hist_image, 8)
    ten = metrics.tenengrad_criterion(hist_image, ksize=3)
    metrix_results = {"ambe": ambe, "de": de, "eme": eme, "ten": ten}
    print(metrix_results)
