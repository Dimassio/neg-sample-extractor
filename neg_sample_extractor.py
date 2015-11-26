import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
# scimage


def get_alpha(mu, sigma):
    # todo: return exp(N(mu, sigma^2))
    return mu


def get_rand_positions(max_y, max_x, width, height):
    # todo: return rand(0, max_y - height), rand(0, max_x - width)
    return 2, 3


def make_negative_samples_rand_states(image, height, N):
    extracted_samples = []
    bin_levels = []
    alpha = get_alpha(0.5, 0.1)
    width = height * alpha
    for iter in range(N):
        i, j = get_rand_positions(len(image), len(image[0]), width, height)
        # todo: extraction
        # todo: extracted_samples.append(...)
        # todo: bin_levels.append(...)
    return extracted_samples, bin_levels


def binarization(image, bin_level):
    binary_image = []
    for i in range(len(image)):
        vec = []
        for j in range(len(image[i])):
            if image[i][j] >= bin_level:
                vec.append(0)
            else:
                vec.append(255)
        binary_image.append(vec)
    return binary_image


# 1. 0.839478641341 - part of white pixels
# 2. part of white and black pixels must be approximately equal
def get_bin_level(image):
    level = 1 # todo: bin search?
    pixels = [0 for i in range(0, 256)]
    for line in image:
        for pixel in line:
            pixels[pixel] += 1
    done = False
    while not done:
        white = 0
        black = 0
        for i in range(level):
            white += pixels[i]
        for i in range(level, 256):
            black += pixels[i]
        if abs(black - white) < len(image):
            done = True
        else:
            level += 1
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
    plt.imshow(image, cmap='Greys')
    plt.show()


def fread(path):
    return img.imread(path)


def fsave(image_list):
    for i in range(len(image_list)):
        img.imsave("NegSamples/" + str(i) + "_neg.png", np.array(image_list[i]), cmap='Greys')

