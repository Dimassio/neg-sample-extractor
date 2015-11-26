import neg_sample_extractor as nse

image = nse.fread('21.tif')

# 20 samples
# 50 is height
neg_samples, bin_levels = nse.make_negative_samples_rand_states(image, 50, 20)

neg_samples_binary = []
for i, sample in enumerate(neg_samples):
    nse.show(sample) # todo: make them without white strips at edges!
    #neg_samples_binary.append(nse.binarization(sample, bin_levels[i]))

#nse.fsave(neg_samples_binary)
