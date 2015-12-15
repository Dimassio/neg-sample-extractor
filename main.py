import neg_sample_extractor as nse

nse.convert_to_jpg('22.tif')
image = nse.fread('22.jpg')
nse.show(image)
# 50 is height
# 10 samples
# 0.5 - mu1
# 0.1 - sigma1
neg_samples, bin_levels = nse.make_negative_samples_rand_states(image, 55, 10, 0.5, 0.1, -0.5, 0.1)
neg_samples_binary = []
for i, sample in enumerate(neg_samples):
    nse.show(sample)
    neg_samples_binary.append(nse.trim(nse.binarization(sample, bin_levels[i])))
    nse.show(neg_samples_binary[-1])

nse.fsave(neg_samples_binary)
