import os.path

import neg_sample_extractor as nse

# 61 120-positive
# 975 - pictures

number_of_samples = 63

elongations = nse.get_positive_elongations("Positive/")
'''
nse.print_elong_diagram(elongations)
'''

for iter in range(1012):

    print "Now " + str(iter) + " image"

    file_path = "Negative/" + str(iter) + ".tif"
    if not os.path.exists(file_path):
        continue
    nse.convert_to_jpg(file_path, "Jpg/" + str(iter) + ".jpg")
    image = nse.fread("Jpg/" + str(iter) + ".jpg")
    neg_samples, bin_levels = nse.make_negative_samples_rand_states(image, 55, number_of_samples, elongations)
    neg_samples_binary = []
    # binarization:
    for i, sample in enumerate(neg_samples):
        neg_samples_binary.append(nse.trim(nse.binarization(sample, bin_levels[i])))
    nse.fsave(neg_samples_binary, iter)  # saving real negative sample


# P.P.S: calculate average height white / all pixels average for all positive samples! and replace everuwhere this number
