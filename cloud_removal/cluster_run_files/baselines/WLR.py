
import numpy as np
import matplotlib.pyplot as plt

import math

# https://www.sciencedirect.com/science/article/pii/S0034425712004786


def find_similar_pixels(features, x, y, b, N_R=20, init_window_size=5, max_window_size=17):
    window_size = init_window_size

    xs = []
    ys = []

    while(len(xs) < N_R):
        if(window_size > max_window_size):
            if(len(xs) == 0):
                # Paper didn't specify what to do in case of length 0; for now just return all in window like in NSPI
                j = 0
                for x2 in range(Tm_x_min, Tm_x_max):
                    for y2 in range(Tm_y_min, Tm_y_max):
                        xs.append(x2)
                        ys.append(y2)
                        j += 1
            window_size -= 2 # Because we didn't actually use it
            break

        xs = []
        ys = []

        Tm_x_min = max(x-window_size, 0)
        Tm_x_max = min(x+window_size, features.shape[1])
        Tm_y_min = max(y-window_size, 0)
        Tm_y_max = min(y+window_size, features.shape[0])
        mu = np.mean(features[Tm_y_min:Tm_y_max, Tm_x_min:Tm_x_max, b]) # Scalar for this band

        T_matrix = np.absolute(features[:,:,b] - mu) # individual T per pixel and band given by T[y,x,b], or band vector as T[y,x,:]

        for x2 in range(Tm_x_min, Tm_x_max):
            for y2 in range(Tm_y_min, Tm_y_max):
                if(not(x2 == x) and not(y2 == y)):
                    val = np.absolute(features[y2, x2, b] - features[y, x, b])
                    if(val < np.sum(T_matrix[y2, x2])):
                        xs.append(x2)
                        ys.append(y2)

        window_size += 2

    assert len(xs) == len(ys)

    return(np.array(xs, dtype=np.int16), np.array(ys, dtype=np.int16), window_size)


def compute_difference_indices(features, x, y, b, xs, ys, alpha=0.001):
    Dis = []
    for i in range(0, len(xs)):
        x2 = xs[i]
        y2 = ys[i]
        t1 = np.absolute(features[y2, x2, b] - features[y, x, b]) + alpha # This is still a vector per band, again paper not clear on this
        t2 = (x2 - x)**2 + (y2 - y)**2
        Di = t1 * t2
        Dis.append(Di)
    Dis = np.array(Dis)
    Wis = (1/Dis) / np.sum(1/Dis)

    return(Wis)


def compute_a_b(input_image, features, x, y, b, xs, ys, Wis, window_size):
    n = len(xs)

    # Not computing Tm anymore but it's using the same window
    Tm_x_min = max(x-window_size, 0)
    Tm_x_max = min(x+window_size, features.shape[1])
    Tm_y_min = max(y-window_size, 0)
    Tm_y_max = min(y+window_size, features.shape[0])
    local_input_image = input_image[Tm_y_min:Tm_y_max, Tm_x_min:Tm_x_max, b]
    local_feature_image = features[Tm_y_min:Tm_y_max, Tm_x_min:Tm_x_max, b]
    lpi_mean = np.mean(local_input_image)
    lfi_mean = np.mean(local_feature_image)

    # For exceptional cases with fewer than 2 pixels, use Eq 8
    if(n < 2):
        a = lpi_mean / lfi_mean
        b_coef = 0
        return(a, b_coef)        

    # The following can probably be vectorised if code runs slowly
    t1 = 0
    for i in range(0, n):
        Wi = Wis[i]
        pi = input_image[ys[i], xs[i], b]
        fi = features[ys[i], xs[i], b]
        t1 += Wi * abs(pi - lpi_mean) * abs(fi - lfi_mean) # TODO: abs was not in the paper, but got weird results without it, if doesn't fix anything get rid of it again
    t2 = 0
    for i in range(0, n):
        Wi = Wis[i]
        fi = features[ys[i], xs[i], b]
        t2 += Wi * (fi - lfi_mean)**2

    #print("t1", t1)
    #print("t2", t2)
    a = t1 / t2

    b_coef = lpi_mean - a * lfi_mean
    return(a, b_coef)


def WLR(input_img, features, mask, cloud_threshold=20, N_R=20, init_window_size=5, max_window_size=17, alpha=0.001):
    res = input_img.copy()

    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            if(mask[i, j] >= cloud_threshold):
                for b in range(0, res.shape[2]):
                    xs, ys, window_size = find_similar_pixels(features, j, i, b, N_R=N_R, init_window_size=init_window_size, max_window_size=max_window_size)
                    Wis = compute_difference_indices(features, j, i, b, xs, ys, alpha=alpha)
                    a, b_coef = compute_a_b(input_img, features, j, i, b, xs, ys, Wis, window_size)

                    #print("a", a)
                    #print("b", b_coef)
                    #print("f", features[i, j, b])
                    #print("\n")

                    res[i, j, b] = a * features[i, j, b] + b_coef
            else:
                res[i, j, :] = input_img[i, j, :]

    return(res)
    
    
    
    