import os.path

import neg_sample_extractor as nse

# 61 120-positive
# 975 - pictures

number_of_samples = 63
number_of_positive_samples = 61120

elongations = nse.get_positive_elongations("Positive/")
'''
nse.print_elong_diagram(elongations)
'''

# todo: calculate white / all pixels average for all positive samples! and replace everuwhere this number
# P.P.S: calculate average height

for iter in range(1012):

    print "Now " + str(iter) + " image"

    file_path = "Negative/" + str(iter) + ".tif"
    if not os.path.exists(file_path):
        continue
    nse.convert_to_jpg(file_path, "Jpg/" + str(iter) + ".jpg")
    image = nse.fread("Jpg/" + str(iter) + ".jpg")
    if iter % 100 == 0:
        nse.show(image)
    neg_samples, bin_levels = nse.make_negative_samples_rand_states(image, 55, number_of_samples, elongations)
    neg_samples_binary = []
    # binarization:
    for i, sample in enumerate(neg_samples):
        # nse.show(sample)
        neg_samples_binary.append(nse.trim(nse.binarization(sample, bin_levels[i])))
        # nse.show(neg_samples_binary[-1])
    nse.fsave(neg_samples_binary, iter)  # saving real negative sample
