import os.path

import neg_sample_extractor as nse

# 49743 of positive
# 954 of negative

number_of_samples = 52

for iter in range(1012):

    print "Now " + str(iter) + " image"

    file_path = "Done/Negative/" + str(iter) + ".tif"
    if not os.path.exists(file_path):
        continue
    nse.convert_to_jpg(file_path, "Done/Jpg/" + str(iter) + ".jpg")
    image = nse.fread("Done/Jpg/" + str(iter) + ".jpg")
    # nse.show(image)
    neg_samples, bin_levels = nse.make_negative_samples_rand_states(image, 55, number_of_samples, 0.5, 0.1, -0.5, 0.1)
    neg_samples_binary = []
    for i, sample in enumerate(neg_samples):
        # nse.show(sample)
        neg_samples_binary.append(nse.trim(nse.binarization(sample, bin_levels[i])))
        # nse.show(neg_samples_binary[-1])
    nse.fsave(neg_samples_binary, iter)
