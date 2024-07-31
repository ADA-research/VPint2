
import numpy as np
import matplotlib.pyplot as plt

import math






# NSPI
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6095313
# https://www.sciencedirect.com/science/article/pii/S0034425710003482


def spectro_spatial_pred(img, b, xs, ys, w):
    tot = 0
    for i in range(0, len(xs)):
        tot += w[i] * img[ys[i], xs[i], b]
    return(tot)


def spectro_temporal_pred(img, img2, x, y, b, xs, ys, w):
    tot = img2[y, x, b]
    for i in range(0, len(xs)):
        tot += w[i] * (img[ys[i], xs[i], b] - img2[ys[i], xs[i], b])
    return(tot)


def compute_weight(img2, x, y, b, xs, ys):
    N = len(xs)
    uds = np.zeros(N)
    rmsds = np.zeros(N)
    for i in range(0, N):
        uds[i] = abs(x - xs[i]) + abs(y - ys[i])
        rmsds[i] = RMSD(img2, x, y, xs[i], ys[i])

    # Normalise for CR
    uds = (uds - np.min(uds)) / (np.max(uds) - np.min(uds)) + 1
    rmsds = (rmsds - np.min(rmsds)) / (np.max(rmsds) - np.min(rmsds)) + 1

    #t2 = np.sum(1 / (uds*rmsds))
    #w = (1 / uds*rmsds) / t2

    cds = uds*rmsds
    t1 = 1 / cds # Vector with individual pixels
    t2 = np.sum(1/cds) # Scalar over all pixels
    w = t1 / t2
    
    return(w)


def RMSD(img2, x, y, x2, y2):
    k = img2.shape[2]
    t1 = 0
    for b in range(0, k):
        t1 += (img2[y2, x2, b] - img2[y, x, b])**2
    rmsd = math.sqrt(t1 / k)
    return(rmsd)


def RMSD_fullimg(img2, x, y):
    k = img2.shape[2]
    rmsd_full = np.zeros_like(img2[:,:,0])
    for b in range(0, k):
        rmsd_full = rmsd_full + (img2[:,:,b] - img2[y, x, b])**2
    rmsd_full = np.sqrt(rmsd_full / k)
    return(rmsd_full)


def compute_r1(shp, x, y, xs, ys):
    # Return average distance between target and similar pixels
    dists = np.zeros(len(xs))
    for i in range(0, len(xs)):
        dists[i] = abs(x-xs[i]) + abs(y-ys[i])
    return(np.mean(dists))


def compute_r2(mask, x, y, threshold):
    # Return spatial distance between target and cloud centre
    max_up = 0
    for y2 in range(0, y):
        if(not(mask[y-y2, x] > threshold)):
            max_up = y2
            break
    max_down = 0
    for y2 in range(0, mask.shape[0]-y):
        if(not(mask[y+y2, x] > threshold)):
            max_down = y2
            break
    max_left = 0
    for x2 in range(0, x):
        if(not(mask[y, x-x2] > threshold)):
            max_left = x2
            break
    max_right = 0
    for x2 in range(0, mask.shape[1]-x):
        if(not(mask[y, x+x2] > threshold)):
            max_right = x2
            break
    centre_x = int((max_right-max_left) / 2)
    centre_y = int((max_right-max_left) / 2)
    r2 = abs(x-centre_x) + abs(y-centre_y)
    return(r2)
    

def compute_r1_orig(img2, x, y, xs, ys):
    N = len(xs)
    B = img2.shape[2]

    mat = img2[ys, xs, :]
    tot2 = np.sqrt(np.sum((mat - img2[y, x, :])**2, axis=1) / B)
    tot = np.sum(tot2)

    r1 = 1/N * tot
    return(r1)


def compute_r2_orig(img, img2, xs, ys):
    N = len(xs)
    B = img2.shape[2]
    tot = 0 # Pixels

    mat = img[ys, xs, :]
    mat2 = img2[ys, xs, :]
    tot2 = np.sqrt(np.sum((mat2 - mat)**2, axis=1) / B)
    tot = np.sum(tot2)

    r2 = 1/N * tot
    return(r2)


def identify_common_pixels(img2, x, y, mask, threshold, stds, N, m=5, max_tries=10000, max_window=17):
    # Find xs, ys
    # Test approach: random sampling
    xs = []
    ys = []

    #rmsd_full = RMSD_fullimg(img2, x, y)


    

    # This full sort is probably inefficient

    IWS = int(((math.sqrt(N) + 1) / 2) * 2 + 1)

    i = 0 # Counter for pixels so far
    while(i < N and IWS <= max_window):
        # Add pixels in moving window, but not in mask and with enough rmsd
        # If not enough are found before max window, just use what we have
        # If no pixels at all, just use all common pixels
        min_x = max(0, x-IWS)
        max_x = min(img2.shape[1]-1, x+IWS)
        min_y = max(0, y-IWS)
        max_y = min(img2.shape[0]-1, y+IWS)
        for x2 in range(min_x, max_x):
            for y2 in range(min_y, max_y):
                if(not(x2 == x and y2 == y)):
                    if(not(mask[y2, x2] > threshold)):
                        rmsd = RMSD(img2, x, y, x2, y2)
                        cond = np.mean([stds[b] * 2 / m for b in range(0, img2.shape[2])])
                        if(rmsd < cond):
                            xs.append(x2)
                            ys.append(y2)
                            i += 1

        IWS += 2 

    if(i == 0):
        j = 0
        for x2 in range(min_x, max_x):
            for y2 in range(min_y, max_y):
                xs.append(x2)
                ys.append(y2)
                j += 1

        #xs = xs[:j]
        #ys = ys[:j]

    #else:
        #xs = xs[:i]
        #ys = ys[:i]

    xs = np.array(xs, dtype=np.int16)
    ys = np.array(ys, dtype=np.int16)

    return(xs, ys)


def NSPI(img, img2, mask, threshold=0.2, N=10, m=5):

    stds = {b:np.mean(img2[:,:,b]) for b in range(0, img2.shape[2])}

    # Run
    res = np.zeros_like(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if(mask[i,j] > threshold):
                xs, ys = identify_common_pixels(img2, j, i, mask, threshold, stds, N, m=m)
                r1 = compute_r1_orig(img2, j, i, xs, ys)
                r2 = compute_r2_orig(img, img2, xs, ys)

                t1 = (1 / r1) / (1/r1 + 1/r2)
                t2 = (1 / r2) / (1/r1 + 1/r2)

                for b in range(0, img.shape[2]):
                    # Actually compute NSPI
                    w = compute_weight(img2, j, i, b, xs, ys)
                    l = t1 * spectro_spatial_pred(img, b, xs, ys, w)
                    l = l + t2 * spectro_temporal_pred(img, img2, j, i, b, xs, ys, w)
                    res[i, j, b] = l
            else:
                res[i, j, :] = img2[i, j, :]

    return(res)
    




