# Haar Transform and Inverse H.T.
# 2021-6-7
# TBSI, THU
# Hengxi Zhang

import numpy as np
# from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image
import os
import time


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
    :return: The result of Haar transformation & the Haar matrix & the detail coefficient matrix
    '''

    M = np.shape(img)[0]
    N = np.shape(img)[1]

    img_HT_row = np.zeros((M, N))
    img_HT = np.zeros((M, N))

    for m in range(M):
        img_HT_row[m, :] = myhaar(img[m, :], N, level_num)  # Haar Transform for each row at first
    for n in range(N):
        img_HT[:, n] = myhaar(img_HT_row[:, n], M, level_num)  # Haar Transform for each column secondly

    # Calculate the average of the detail coefficients
    M_section = int(M / (2 ** level_num))
    N_section = int(N / (2 ** level_num))
    
    detail_coeff_of_level_num = img_HT[M_section:(2*M_section),N_section:(2*N_section)] 
    abs_detail_coeff_of_level_num = np.abs(detail_coeff_of_level_num)
    
    img_HT_Haar_mat = img_HT[:M_section, :N_section]  # Haar coefficient matrix, i.e. compressed image

    return img_HT, img_HT_Haar_mat, abs_detail_coeff_of_level_num


if __name__ == '__main__':

    tic = time.time()

    # pos
    pos_directory_name = r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_PA\inria\pos'
    pos_sum_collector = np.zeros((32,16))
    pos_num = 0

    for filename in os.listdir(pos_directory_name):

        image = Image.open(pos_directory_name +"\\"+ filename)
        image_gray = image.convert('L')  # Convert the colorful image into gray
        image_gray_array = np.array(image_gray)


        level_num = 2
        image_HT, image_HT_Haar, detail_coeff_max_of_level_num = myhaar2(image_gray_array, level_num)
        pos_sum_collector += detail_coeff_max_of_level_num
        pos_num += 1
        print('pos_num',pos_num)

    pos_detail_coeff_mat_ave = pos_sum_collector / pos_num
    print('pos:', pos_sum_collector)

    # neg
    neg_sum_collector = np.zeros((32,16))
    neg_num = 0
    neg_directory_name = r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_PA\inria\neg'
    for filename in os.listdir(neg_directory_name):

        image = Image.open(neg_directory_name +"\\"+ filename)
        image_gray = image.convert('L')  # Convert the colorful image into gray
        image_gray_array = np.array(image_gray)


        level_num = 2
        image_HT, image_HT_Haar, detail_coeff_max_of_level_num = myhaar2(image_gray_array, level_num)
        neg_sum_collector += detail_coeff_max_of_level_num
        neg_num += 1
        print('neg_num', neg_num)

    neg_detail_coeff_mat_ave = neg_sum_collector / neg_num
    print('neg:', neg_detail_coeff_mat_ave)


    # # detail coefficients
    # # image = Image.open(r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_PA\inria\pos\crop_000010a.png')
    # image = Image.open(
    #     r'F:\清华大学TBSI\2 Spring 2021\1 Courses\Advanced Signal Processing\ASP_HW\ASP_PA\inria\neg\1.png')
    # image_gray = image.convert('L')  # Convert the colorful image into gray
    # image_gray_array = np.array(image_gray)
    #
    # level_num = 2
    # image_HT, image_HT_Haar, detail_coeff_max_of_level_num = myhaar2(image_gray_array, level_num)

    print('%d pos image patches are calculated.'% pos_num)
    print('%d neg image patches are calculated.' % neg_num)
    toc = time.time()
    T = toc - tic
    print('Time:', T, str('s'))

    plt.subplot(1, 2, 1)
    plt.imshow(pos_detail_coeff_mat_ave, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(neg_detail_coeff_mat_ave, cmap='gray')
    plt.show()