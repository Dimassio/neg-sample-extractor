import os.path

import neg_sample_extractor as nse

# 61 120-positive
# 975 - pictures

number_of_samples = 70
number_of_pictures = 1000
average_height = 41  # 40.9546581887

elongations, pixel_percentages = nse.get_positive_elongations("../Data/Positive/")

for iter in range(number_of_pictures):

    print "Now " + str(iter) + " image"

    file_path = "Negative/" + str(iter) + ".tif"
    if not os.path.exists(file_path):
        continue
    nse.convert_to_jpg(file_path, "Jpg/" + str(iter) + ".jpg")
    image = nse.fread("Jpg/" + str(iter) + ".jpg")
    neg_samples, bin_levels = nse.make_negative_samples_rand_states(image, average_height, number_of_samples,
                                                                    elongations, pixel_percentages)
    # binarization
    neg_samples_binary = []
    for i, sample in enumerate(neg_samples):
        neg_samples_binary.append(nse.trim(nse.binarization(sample, bin_levels[i])))
    nse.fsave(neg_samples_binary, iter)  # saving real negative sample

