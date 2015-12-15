import math
import os

import numpy as np
import numpy.random as rd
from PIL import Image
from scipy import transpose
from skimage import io
from sklearn import mixture


def convert_to_jpg(path):
    outfile = os.path.splitext(path)[0] + ".jpg"
    if path != outfile:
        try:
            Image.open(path).save(outfile)
        except IOError:
            print "cannot convert", path


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


def get_gmm_model(mu1, sigma1, mu2, sigma2):
    np.random.seed()
    mix_model = mixture.GMM(n_components=2)
    obs = np.concatenate((sigma1 * np.random.randn(300, 1) + mu1, sigma2 * np.random.randn(300, 1) + mu2))
    mix_model.fit(obs)
    return mix_model


def get_alpha(model):
    return model.sample()


def get_rand_positions(max_y, max_x, width, height):
    return rd.randint(0, max_y - height), rd.randint(0, max_x - width)


def make_negative_samples_rand_states(image, height, N, mu1, sigma1, mu2, sigma2):
    extracted_samples = []
    bin_levels = []
    model = get_gmm_model(mu1, sigma1, mu2, sigma2)
    alpha = math.exp(get_alpha(model))
    width = height * alpha
    iter = 0
    while iter < N:
        i, j = get_rand_positions(len(image), len(image[0]), width, height)
        layer = image[i:i + height]
        block = [layer[q][j:j + width] for q in range(len(layer))]
        level = get_bin_level(block)
        if level == -1:  # if level, that we needed between black and white pixels not found
            continue
        else:
            iter += 1
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
    level = 1 # todo: bin search?
    pixels = [0 for i in range(0, 256)]
    for line in image:
        for pixel in line:
            pixels[pixel] += 1
    done = False
    while not done and level <= 255:
        white = 0.0
        black = 0.0
        for i in range(level):
            white += pixels[i]
        for i in range(level, 256):
            black += pixels[i]
        if abs(black / (white + black) - 0.5) < 0.01:
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


# No random extraction of negative samples with fixed width and height as a parameters
def make_negative_samples_fix_width(image, height, min_width=100):
    y_size = len(image)
    x_size = len(image[0])
    if height > y_size or height <= 0:
        print "nse:make negative samples failed: wrong height"
        return

    i = 0
    extracted_images = []
    while i + height < y_size:  # block on 'height' pixels high | i -> i + height
        for k in range(0, x_size, 100):  # offset from beginning of the block | k
            for j in range(k + 1, x_size):  # block on j pixels width | k -> j + 1
                if j + 1 - k < min_width:  # never less then minimum width
                    continue
                extracted_image_block = image[i: i + height]
                block = [extracted_image_block[q][k: j + 1] for q in range(len(extracted_image_block))]
                if abs(get_pixel_percentage(block) - 0.839478641341) < 0.05:
                    extracted_images.append(block)
                    break
            print "Size of generated samples = " + str(len(extracted_images))
        i += 10
    return extracted_images


def show(image):
    io.imshow(np.array(image))
    io.show()


def fread(path):
    return io.imread(path)


def fsave(image_list):
    for i in range(len(image_list)):
        io.imsave("NegSamples/" + str(i) + "_neg.png", np.array(image_list[i]))

