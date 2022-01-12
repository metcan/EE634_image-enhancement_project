# Import libraries
import logging
import os
import re
from threading import local
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from numpy.core.numeric import zeros_like 
from itertools import permutations
from numba import njit
import sys
from time import process_time
IMSHOW_DIVIDER = 4
ATTENUATION_FACTOR = 9
epsilon = np.finfo(np.double).eps

@njit
def get_local_min(image, kernel_size=7):
    movements = [[row - kernel_size//2, column- kernel_size//2] for row in range(0, kernel_size, 1) for column in range(0, kernel_size, 1)]
    local_min = np.ones_like(image) 
    max_y, max_x = local_min.shape
    for iy, ix in np.ndindex(local_min.shape):
        min = 2^32
        for y, x in movements:
            ny = iy + y
            nx = ix + x
            if (max_y>ny>=0 and max_x>nx>=0):
                if (min > image[ny][nx]):
                    min = image[ny][nx]
        local_min[iy, ix] =  min
    return local_min

@njit
def get_local_max(image, kernel_size=7):
    movements = [[row - kernel_size//2, column - kernel_size//2] for row in range(0, kernel_size, 1) for column in range(0, kernel_size, 1)]
    local_max = np.zeros_like(image)
    max_y, max_x = local_max.shape
    for iy, ix in np.ndindex(local_max.shape):
        max = 0
        for y, x in movements:
            ny = iy + y
            nx = ix + x
            if (max_y>ny>=0 and max_x>nx>=0):
                if (max < image[ny][nx]):
                    max = image[ny][nx]
        local_max[iy, ix] =  max
    return local_max

def load_img(data_path = ""):
    return cv2.imread(data_path,0)

def histogram_equlization(img):
    return cv2.equalizeHist(img)

def normalize_image(img):
    return img / np.max(img)

def merge_images(images):
    res = np.hstack(images)

def display_images(images):
    lenght = len(images)
    x = 2
    y = lenght // 2
    f, axarr = plt.subplots(y,x)
    for index, image in enumerate(images):
        j = index // 2
        i = index % 2
        axarr[j,i].imshow(image*255, cmap='gray', vmin=0, vmax=255)
    plt.show()

def show_image(img, window_name= "window"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
# Generate E_i
# Merge E_i to obtain F 

def calculate_lambda(attenuation_factor, image, local_min, local_max):
    _local_max = local_max + epsilon
    quotient = 1.0 - (attenuation_factor * local_min * (1.0 / _local_max - 1.0) )
    quotient = np.log(quotient)
    divisor = np.log(local_max)
    divisor = np.nan_to_num(divisor) + epsilon
    result = quotient / divisor
    result = np.nan_to_num(result)
    return result

def apply_gaussian_filter(image, kernel_size = 5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_laplacian_filter(image, kernel_size = 5):
    ddepth = cv2.CV_64FC1 # integer scaling
    return np.abs(cv2.Laplacian(image, ddepth = ddepth, ksize = kernel_size))


def calculate_enhanced_image(attenuation_factor, image, lambda_array,local_min, local_max):
    quotient = (image - attenuation_factor*local_min)
    divisor = (np.power(local_max, lambda_array) - attenuation_factor*local_min) 
    result = quotient / divisor
    result = np.nan_to_num(result)
    return result

def calculate_constrast_level(image, kernel_size=5, epsilon=sys.float_info.epsilon):
    return apply_laplacian_filter(image, kernel_size) / (apply_gaussian_filter(image, kernel_size) + epsilon)

def calculate_brightness_preservation_metric(image, local_max):
    return np.exp(-( (image - local_max) / np.std(local_max) )**2)

if __name__ == "__main__":
    data_path = "chest.png"
    image = load_img(data_path)
    Image_Size = (512, 512)
    image = cv2.resize(image, Image_Size, interpolation = cv2.INTER_AREA)
    image = histogram_equlization(image)
    image = normalize_image(image)
    local_min = get_local_min(image, kernel_size=7)
    local_max = get_local_max(image, kernel_size=7)
    K = np.linspace(0, 1, ATTENUATION_FACTOR)
    indexlist = np.where(local_max==0)
    indexlisttranspose = np.array(indexlist).T
    
    weight_matrix_list = []
    enhanced_image_list = []
    for i in K:
        lambda_array = calculate_lambda(i, image, local_min, local_max)
        if (len(indexlisttranspose) !=0):
            lambda_array[ indexlisttranspose[:,0],indexlisttranspose[:,1]] = 0
        enhanced_image = calculate_enhanced_image(i, image, lambda_array,local_min, local_max)
        if (len(indexlisttranspose) !=0):
            enhanced_image[ indexlisttranspose[:,0],indexlisttranspose[:,1]] = 0
        enhanced_image_list.append(enhanced_image)
        contrast_level = calculate_constrast_level(enhanced_image)
        brightness_preservation_metric = calculate_brightness_preservation_metric(enhanced_image, local_max)
        weight_matrix = brightness_preservation_metric * contrast_level
        weight_matrix_list.append(weight_matrix)
        print(f"Lambda array max value = {np.max(lambda_array)} and min value = {np.min(lambda_array)}")
        print(f"Contrast Level array max value = {np.max(contrast_level)} and min value = {np.min(contrast_level)}")
        print(f"Weight array max value = {np.max(weight_matrix)} and min value = {np.min(weight_matrix)}")
    weight_sum_inv = (np.sum(weight_matrix_list, axis=0) + epsilon)**-1
    for i in range(len(weight_matrix_list)):
        weight_matrix_list[i] = weight_sum_inv * weight_matrix_list[i]
    result = (np.sum([weight *image for weight, image in zip(weight_matrix_list, enhanced_image_list)], axis=0))
    normalizedImg = np.zeros_like(result)
    normalizedImg = np.uint8(cv2.normalize(result,  normalizedImg, 0, 255, cv2.NORM_MINMAX))
    #normalizedImg = np.uint8(result* 255)
    show_image(normalizedImg)
    print(result)