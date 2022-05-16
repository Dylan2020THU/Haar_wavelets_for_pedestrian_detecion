# Haar Transform and Inverse H.T.
# 2021-6-7
# TBSI, THU
# Hengxi Zhang

import numpy as np
# from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image
# import copy
# import pickle


def myhaar(x, L, level_num):
    '''
    :param x: The original input signal
    :param L: The length of the input signal
    :param level_num: The transformation times
    :return: The result of the Haar Transformation
    '''

    mid = int(L / 2)
    x_haar = np.zeros(L)  # Used to store the result of Haar Transformation

    for num in range(level_num):
        ave = np.zeros(mid)
        diff = np.zeros(mid)
        for i in range(mid):
            ave[i] = (x[2 * i] + x[2 * i + 1]) / 2  # The front part of the Haar transformation
            diff[i] = x[2 * i] - ave[i]  # The latter part

        x_haar[:mid] = ave
        x_haar[mid:(2 * mid)] = diff

        x = x_haar[:mid]  # Take the front part as the object for next iteration
        # print('mid:',mid)
        mid = int(mid / 2)

    return x_haar


def myihaar(w, L, level_num):
    '''
    :param w: The original input signal
    :param L: The length of the input signal
    :param level_num: The transformation times
    :return: The result of the inverse Haar Transformation
    '''

    # level_num = int(np.log2(L))   # The biggest transformation times
    x_ihaar = np.zeros(L)
    mid = int(L / 2**level_num)  # Verify the midline
    for num in range(level_num):
        if mid <= int(L / 2):
            ave = w[:mid]  # Pick the front half part
            diff = w[mid:(2 * mid)]  # Pick the latter half part
            for i in range(mid):
                x_ihaar[2 * i] = ave[i] + diff[i]
                x_ihaar[2 * i + 1] = ave[i] - diff[i]
            w[:(2 * mid)] = x_ihaar[:(2 * mid)]
            mid = mid * 2
    # test = x_ihaar - w
    # print('test', test)
    return x_ihaar


def myhaar2(img, level_num):
    '''
    :param img:  The original input image
    :param level_num: The transformation times
    :return: The result of Haar transformation & the average of absolute detail coefficients
    '''

    M = np.shape(img)[0]
    N = np.shape(img)[1]

    img_HT_row = np.zeros((M, N))
    img_HT = np.zeros((M, N))

    for m in range(M):
        img_HT_row[m, :] = myhaar(img[m, :], N, level_num)  # Haar Transform for each row at first
    for n in range(N):
        img_HT[:, n] = myhaar(img_HT_row[:, n], M, level_num)  # Haar Transform for each row secondly

    # Calculate the average of the detail coefficients
    M_section = int(M / (2 ** level_num))
    N_section = int(N / (2 ** level_num))
    detail_coeff_mat_of_level_num = img_HT[M_section:(2 * M_section), N_section:(2 * N_section)]
    img_HT_Haar_mat = img_HT[:M_section, :N_section]  # Haar coefficient matrix, i.e. compressed image
    abs_detail_coeff_of_level_num = np.abs(detail_coeff_mat_of_level_num)

    return img_HT, img_HT_Haar_mat, abs_detail_coeff_of_level_num


if __name__ == '__main__':
    # np.random.seed(2021)
    # '''  '''
    # # ------------------ 1-D signal Haar Transformation ------------------ #
    # signal = np.random.randint(0, 10, size=2**10)  # Input signal
    # # signal = np.array([4, 6, 5, 5, 7, 5, 6, 6])
    # L = len(signal)  # The length of the signal
    # signal_HT = myhaar(signal, L, 2)
    # print(signal_HT)
    # signal_recover = myihaar(copy.copy(signal_HT), L, 2)
    # print(signal_recover)
    # delta = signal_recover - signal  # Check the transformation, if each element equals zero, then correct


    # ------------------ 2-D image Haar Transform ------------------ #

    image = Image.open(r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_PA\test_photo\test1.png')
    # image = Image.open(
    #         r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_PA\inria\pos\crop001073a.png')
    image_gray = image.convert('L')  # Convert the colorful image into gray
    image_gray_array = np.array(image_gray)

    # 2-D Haar Transform
    # level_num = int(input("Input the level you want to measure:"))  # The number of the level in Haar Transform
    level_num = 2
    image_HT, image_HT_Haar, detail_coeff_max_of_level_num = myhaar2(image_gray_array, level_num)
    plt.subplot(2, 3, 1)
    plt.imshow(image_gray_array, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.imshow(image_HT, cmap='gray')

    M_sec = int(np.shape(image_HT)[0]/2**level_num)
    N_sec = int(np.shape(image_HT)[1]/2**level_num)
    plt.subplot(2, 3, 3)
    plt.imshow(image_HT_Haar, cmap='gray')


    # Comparison image
    M = np.shape(image)[0]
    N = np.shape(image)[1]

    # Comparison image reading
    # img_cmp = np.random.randint(0,256,size=(M,N))
    # img_cmp = Image.open(
    #     r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_PA\Coding files\beauty.jpg')
    img_cmp = Image.open(
        r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_PA\inria\neg\210.png')
    img_cmp_gray = img_cmp.convert('L')  # Convert the colorful image into gray
    img_cmp_gray_array = np.array(img_cmp_gray)

    # 2-D Haar Transform
    image_cmp_HT, image_cmp_HT_Haar, cmp_detail_coeff_max_of_level_num = myhaar2(img_cmp_gray_array, level_num)
    plt.subplot(2, 3, 4)
    plt.imshow(img_cmp_gray_array, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.imshow(image_cmp_HT, cmap='gray')

    M_sec = int(np.shape(image_cmp_HT)[0]/2**level_num)
    N_sec = int(np.shape(image_cmp_HT)[1]/2**level_num)
    plt.subplot(2, 3, 6)
    plt.imshow(image_cmp_HT_Haar, cmap='gray')

