import neg_sample_extractor as nse
# todo: tif - doesn't work!!!

image = nse.fread('0.png')

# 20 samples
# 50 is height
# 0.5 - mu
# 0.1 - sigma
neg_samples, bin_levels = nse.make_negative_samples_rand_states(image, 50, 10, 0.5, 0.1)
neg_samples_binary = []
for i, sample in enumerate(neg_samples):
    nse.show(sample)
    neg_samples_binary.append(nse.remove_white_edges(nse.binarization(sample, bin_levels[i])))

nse.fsave(neg_samples_binary)
