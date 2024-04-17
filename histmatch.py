import numpy as np


def hist_match(sample_image, ref_hist):
    im_flat = sample_image.reshape(-1, sample_image.shape[-1])

    newim = np.zeros(sample_image.shape, 'uint32')

    Amax = np.max(sample_image)

    M = np.zeros((3, Amax+1), 'uint32')

    for l in range(3): # loop through channels
        chan_flat = im_flat[:,l]
        chan_ref = ref_hist[:,l]

        usemax = np.max(chan_flat)

        useedges = np.linspace(0, usemax+1, usemax+2)

        histcounts, edges = np.histogram(chan_flat, useedges)

        cdf = np.cumsum(histcounts) / chan_flat.size

        sumref = np.cumsum(chan_ref)
        cdf_ref = sumref / np.max(sumref)

        for idx in range(usemax+1):
            ind = np.argmin(np.abs(cdf[idx]-cdf_ref))
            M[l, idx] = ind

        for i in range(sample_image.shape[0]):
            for j in range(sample_image.shape[1]):
                for k in range(sample_image.shape[2]):
                    newim[i,j,k,l] = M[l, sample_image[i,j,k,l]]       
    
    return newim