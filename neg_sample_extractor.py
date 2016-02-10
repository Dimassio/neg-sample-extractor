import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import plotly.plotly as py
from PIL import Image
from scipy import transpose
from skimage import io
from sklearn import mixture

number_of_positive_samples = 61120


def print_elong_diagram(elongations):
    print "printing histogram of elongations..."
    plt.hist(np.array(elongations))
    plt.title("Elongations of positive samples")
    plt.xlabel("Elongation")
    plt.ylabel("Frequency")
    fig = plt.gcf()
    py.sign_in('Dimassio', 'ebukn0pzpq')
    plot_url = py.plot_mpl(fig, filename='elongations-histogram')


def get_positive_elongations(src):
    print "getting elongations..."
    result = []
    counter = 0
    for iter in range(10000):  # number_of_positive_samples):
        if not os.path.exists(src + str(iter) + ".tif"):
            continue
        counter += 1
        print counter
        dest_path = "Temp/" + str(iter) + ".jpg"
        src_path = src + str(iter) + ".tif"
        convert_to_jpg(src_path, dest_path)
        image = fread(dest_path)
        result.append([math.log(float(len(image[0])) / float(len(image)))])  # brackets only for using GMM
    return result


def convert_to_jpg(src, dest):
    if src != dest:
        try:
            Image.open(src).save(dest)
        except IOError:
            print "cannot convert", src


def trim(image):  # 255 white
    tr_image = transpose(image)
    start = 0
    while sum(tr_image[start]) == 255 * len(tr_image[start]):
        # condition on i is not needed, because of the balance between white and black
        start += 1
    finish = len(tr_image) - 1
    while sum(tr_image[finish]) == 255 * len(tr_image[finish]):
        finish -= 1
    return transpose(tr_image[start: finish + 1])


def get_gmm_model(elongations):
    np.random.seed()
    mix_model = mixture.GMM(n_components=1)  # according to histogram
    mix_model.fit(elongations)
    return mix_model


def get_alpha(model):
    return model.sample()


def get_rand_positions(max_y, max_x, width, height):
    return rd.randint(0, max_y - height), rd.randint(0, max_x - width)


def make_negative_samples_rand_states(image, height, N, elongations):
    '''
    :param image: input image
    :param height: height we want
    :param N: number of samples we want
    :param elongations: set of elongations
    :return: extracted samples and their binariation levels
    '''

    extracted_samples = []
    bin_levels = []
    model = get_gmm_model(elongations)
    alpha = math.exp(get_alpha(model))
    width = height * alpha
    iter = 0
    counter = 0
    while iter < N:
        if counter > 500:  # we can't find anything
            break
        if len(image) <= height or len(image[0]) <= width:  # source image too small
            break
        i, j = get_rand_positions(len(image), len(image[0]), width, height)
        layer = image[i:i + height]
        block = [layer[q][j:j + width] for q in range(len(layer))]  # current subsample from picture
        level = get_bin_level(block)
        if level == -1:  # if level, that we needed between black and white pixels not found
            counter += 1
            continue
        else:
            iter += 1  # found subsample
        extracted_samples.append(block)
        bin_levels.append(level)
    return extracted_samples, bin_levels


def binarization(image, bin_level):
    binary_image = []
    for i in range(len(image)):
        vec = []
        for j in range(len(image[i])):
            if image[i][j] >= bin_level:
                vec.append(255)
            else:
                vec.append(0)
        binary_image.append(vec)
    return binary_image


# 1. 0.839478641341 - part of white pixels in image of thai words
# 2. part of white and black pixels must be approximately equal
def get_bin_level(image):
    pixels = [0 for i in range(0, 256)]
    for line in image:
        for pixel in line:
            pixels[pixel] += 1
    done = False
    level = 1
    while not done and level <= 255:
        white = 0.0
        black = 0.0
        for i in range(level):
            white += pixels[i]
        for i in range(level, 256):
            black += pixels[i]
        if abs(float(black) / float(white + black) - 0.839478641341) < 0.05:  # todo: bin level
            done = True
        else:
            level += 1
    if not done:
        return -1
    return level


def get_pixel_percentage(image):
    positive = 0.0
    negative = 0.0
    for layer in image:
        for pixel in layer:
            if pixel == 0:
                negative += 1
            else:
                positive += 1
    return negative / (positive + negative)


def show(image):
    io.imshow(np.array(image))
    io.show()


def fread(path):
    return io.imread(path)


def fsave(image_list, num):
    for i in range(len(image_list)):
        io.imsave("Pictures/" + str(num) + "_" + str(i) + ".png", np.array(image_list[i]))
