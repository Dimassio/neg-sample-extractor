import neg_sample_extractor as nse

image = nse.fread('0.png')

# 50 is height
# 10 samples
# 0.5 - mu
# 0.1 - sigma
neg_samples, bin_levels = nse.make_negative_samples_rand_states(image, 50, 10, 0.5, 0.1)
neg_samples_binary = []
for i, sample in enumerate(neg_samples):
    neg_samples_binary.append(nse.trim(nse.binarization(sample, bin_levels[i])))
    nse.show(neg_samples_binary[-1])

nse.fsave(neg_samples_binary)
