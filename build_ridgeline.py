import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from image import plot as iplot
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import circmean, circstd
import sys
sys.path.insert(0, '/home/rtodorov/jetpol/ve/vlbi_errors')
from utils import fit_gaussian, gaussian, find_bbox, find_image_std


def get_distance(r1, r2, phi1, phi2):
    return np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(phi1-phi2))


def circular_mean(angles, w=None):
    angles = np.array(angles)
    if w is None:
        w = np.ones(angles.size)
    x = y = 0.
    for angle, weight in zip(angles, w):
        x += np.cos(angle) * weight
        y += np.sin(angle) * weight

    mean = np.arctan2(y, x)
    return mean


def normalize(arr):
    arr = np.array(arr)
    return arr/np.sum(arr)


def get_appr_core_radius(img, mapsize, noise):
    core = np.argmax(img)
    x_max, y_max = np.unravel_index(core, img.shape)
    dx = 0
    dy = 0
    while (img[x_max + dx][y_max + dy] > 2 * noise and
           img[x_max + dx][y_max - dy] > 2 * noise and
           img[x_max - dx][y_max + dy] > 2 * noise and
           img[x_max - dx][y_max - dy] > 2 * noise):
        dx += 1
        dy += 1
    return dx


def subtract_gaussian_core(img, mapsize, noise):
    delta = get_appr_core_radius(img, mapsize, noise)
    print(delta)
    x_left = round(mapsize[0] / 2 - delta)
    x_right = round(mapsize[0] / 2 + delta)
    y_left = round(mapsize[1] / 2 - delta)
    y_right = round(mapsize[1] / 2 + delta)
    height, x0, y0, bmaj, e, bpa = fit_gaussian(img[x_left:x_right, y_left:y_right])
    func = gaussian(height, x0, y0, bmaj, e, bpa)
    x = np.linspace(-x_left + x0, x_right - x0, mapsize[0])
    y = np.linspace(-y_left + y0, y_right - y0, mapsize[1])
    xs, ys = np.meshgrid(x, y)
    # plt.imshow(func(xs, ys))
    # plt.imshow(img - func(ys, xs))
    mask = func(ys, xs) < 4 * noise
    return img * mask


def subtract_core(img, beam, mapsize, noise):
    delta = get_appr_core_radius(img, mapsize, noise)
    print(delta)
    x_left = round(mapsize[0] / 2 - delta)
    x_right = round(mapsize[0] / 2 + delta)
    y_left = round(mapsize[1] / 2 - delta)
    y_right = round(mapsize[1] / 2 + delta)
    core = np.argmax(img)
    x_max, y_max = np.unravel_index(core, img.shape)
    func = gaussian(img[x_max, y_max], x_max - x_left, y_max - y_left, 25., beam[1] / beam[0], beam[2])
    x = np.linspace(-x_left, x_right, mapsize[0])
    y = np.linspace(-y_left, y_right, mapsize[1])
    xs, ys = np.meshgrid(x, y)
    # plt.imshow(func(xs, ys))
    # plt.imshow(img - func(ys, xs))
    mask = func(ys, xs) < 10 * noise
    # plt.imshow(np.log(img * mask + 1))
    # plt.show()
    return img * mask


def get_univariate_spl(x, y, w):
    spl = UnivariateSpline(x, y, w=w, s=0.1)

    i = 1
    while np.isnan(spl(0)):
        try:
            spl = UnivariateSpline(x[::i], y[::i], w=w[::i])
        except:
            pass
        if i > 10:
            return None, None
        i += 1
    # spl.set_smoothing_factor(100)
    return spl


def get_smooth_spl(x, y):
    conv_len = 10
    conv_core = np.ones(2 * conv_len + 1) / (2 * conv_len + 1)
    y[conv_len:-conv_len] = np.convolve(y, conv_core, 'same')[conv_len:-conv_len]
    spl = interp1d(x, y, kind='cubic')
    return spl


def get_smooth_univariate_spl(x, y, w):
    if x.size > 30:
        conv_len = 2
    elif x.size > 15:
        conv_len = 2
    elif x.size > 10:
        conv_len = 1
    else:
        return None
    conv_core = np.ones(2 * conv_len + 1) / (2 * conv_len + 1)

    y[conv_len:-conv_len] = np.convolve(y, conv_core, 'same')[conv_len:-conv_len]
    for i in np.arange(conv_len, 0, -1):
        y[i] = np.mean(y[i:i + conv_len])
        if i > 1:
            y[-i] = np.mean(y[-i - 2:-i + 1])
        else:
            y[-1] = np.mean(y[-3:])
        w[-i] = 2 * (conv_len - i + 1)

    spl = UnivariateSpline(x, y, w=w, s=0.05)

    i = 1
    while np.isnan(spl(0)):
        try:
            spl = UnivariateSpline(x[::i], y[::i], w=w[::i], s=0.1)
        except:
            pass
        if i > 10:
            return None, None
        i += 1
    # spl.set_smoothing_factor(100)
    # plt.plot(x, y)
    # plt.plot(x, spl(x))
    # plt.show()
    return spl


def get_ridgeline(image_data, beam, mapsize, min_abs_level, noise, pix_size):
    npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
    std = find_image_std(image_data, beam_npixels=npixels_beam)
    # image_data_for_ridge = subtract_core(image_data, beam, mapsize, std)
    image_data_for_ridge = image_data
    print("IPOL image std = {} mJy/beam".format(1000 * std))
    print("Noise = {} mJy/beam".format(1000 * noise))

    # locate ridgeline
    core = np.argmax(image_data)
    x_max, y_max = np.unravel_index(core, image_data.shape)
    lmapsize = round(np.hypot(mapsize[0], mapsize[1]))
    lmap = [[0, 0, 0] for _ in range(lmapsize)]
    for x in np.arange(mapsize[0]):
        for y in np.arange(mapsize[1]):
            length = round(np.hypot(x - x_max, y - y_max))
            if lmap[length][2] < image_data_for_ridge[x, y] and 20 * std < image_data_for_ridge[x, y]:
                lmap[length] = [x, y, image_data_for_ridge[x, y]]

    x_bound = np.abs(mapsize[0] * pix_size[0] / 57.3 / 2)
    y_bound = np.abs(mapsize[1] * pix_size[1] / 57.3 / 2)

    ridgeline = [[], [], []]
    ridgeline[0].append(0)
    ridgeline[1].append(0)
    ridgeline[2].append(100)
    core_radius = round(get_appr_core_radius(image_data_for_ridge, mapsize, noise))
    print("Estimated core radius =", core_radius, "pix")
    weight = 15
    for i in np.arange(1, len(lmap)):
        if lmap[i][2] > 0 and i > core_radius:
            ridgeline[0].append((lmap[i][0] - x_max) / mapsize[0] * 2 * x_bound * 206265000)
            ridgeline[1].append((-lmap[i][1] + y_max) / mapsize[1] * 2 * y_bound * 206265000)
            # ridgeline[2].append(max(weight, 1))
            ridgeline[2].append(1)
            weight -= 1
        elif lmap[i][2] > 0 and i < core_radius:
            ridgeline[0].append((lmap[i][0] - x_max) / mapsize[0] * 2 * x_bound * 206265000)
            ridgeline[1].append((-lmap[i][1] + y_max) / mapsize[1] * 2 * y_bound * 206265000)
            ridgeline[2].append(1)

    ridgeline = np.array(ridgeline)
    raw_ridgeline = ridgeline
    # ridgeline_polar - ridgeline points coordinates in polar coordinates
    # ridgeline_polar[0] - radius, ridgeline_polar[1] - theta
    ridgeline_polar = np.zeros(ridgeline.shape)
    for i in np.arange(ridgeline[2].size - 1, -1, -1):
        ridgeline_polar[0][i] = np.hypot(ridgeline[0][i], ridgeline[1][i])
        if ridgeline_polar[0][i] > 0:
            if ridgeline[1][i] >= 0:
                ridgeline_polar[1][i] = np.arcsin(ridgeline[0][i] / ridgeline_polar[0][i])
            else:
                ridgeline_polar[1][i] = np.pi - np.arcsin(ridgeline[0][i] / ridgeline_polar[0][i])
        elif ridgeline_polar[0][i] == 0:
            pass
            # ridgeline_polar[1][i] = ridgeline_polar[1][i+1]
        else:
            raise Exception('Somehow we got negative radius!')

    # shift angles on 2pi
    ridgeline_polar[2] = ridgeline[2]
    ridgeline_polar[ridgeline_polar < 0] += 2 * np.pi
    mean = circmean(ridgeline_polar[1])
    std = circstd(ridgeline_polar[1])
    for i in np.arange(ridgeline_polar[1].size):
        if np.abs(ridgeline_polar[1][i] + 2 * np.pi - mean) < np.abs(ridgeline_polar[1][i] - mean):
            ridgeline_polar[1][i] += 2 * np.pi
        if np.abs(ridgeline_polar[1][i] - 2 * np.pi - mean) < np.abs(ridgeline_polar[1][i] - mean):
            ridgeline_polar[1][i] -= 2 * np.pi
    '''
    # delete too different points
    for i in np.arange(ridgeline_polar[2].size):
        if np.abs(ridgeline_polar[1][i] - mean) > np.pi / 3:
            ridgeline_polar[0][i] = 0

    # get points around the core with mean angle
    for i in np.arange(ridgeline[2].size-2, -1, -1):
        if ridgeline_polar[0][i] == 0:
            ridgeline_polar[1][i] = ridgeline_polar[1][i+1]
        elif ridgeline_polar[2][i] == 0:
            ridgeline_polar[1][i] = circmean(ridgeline_polar[1, i:i+5])
            ridgeline_polar[2][i] = 1
    '''
    # delete too different points
    # get points around the core with mean angle
    for i in np.arange(ridgeline[2].size - 2, -1, -1):
        if ridgeline_polar[0][i] == 0:
            ridgeline_polar[1][i] = ridgeline_polar[1][i + 1]
        if np.abs(ridgeline_polar[1][i] - mean) > std:
            ridgeline_polar[1][i] = circmean(ridgeline_polar[1, i:i + 5])

    for i in np.arange(ridgeline_polar[1].size):
        if np.abs(ridgeline_polar[1][i] + 2 * np.pi - mean) < np.abs(ridgeline_polar[1][i] - mean):
            ridgeline_polar[1][i] += 2 * np.pi
        if np.abs(ridgeline_polar[1][i] - 2 * np.pi - mean) < np.abs(ridgeline_polar[1][i] - mean):
            ridgeline_polar[1][i] -= 2 * np.pi
        if np.isnan(ridgeline_polar[1][i]):
            ridgeline_polar[1][i] = ridgeline_polar[1][i - 1]
    ridgeline_polar = ridgeline_polar[:, ridgeline_polar[0].argsort()]

    # spl = get_univariate_spl(x=ridgeline_polar[0], y=ridgeline_polar[1], w=ridgeline_polar[2])
    spl = get_smooth_univariate_spl(x=ridgeline_polar[0], y=ridgeline_polar[1], w=ridgeline_polar[2])

    return spl, ridgeline_polar, raw_ridgeline


def get_average_ridgeline(image_data, beam, mapsize, min_abs_level, noise, pix_size):
    npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
    std = find_image_std(image_data, beam_npixels=npixels_beam)
    # image_data_for_ridge = subtract_core(image_data, beam, mapsize, std)
    image_data_for_ridge = image_data
    print("IPOL image std = {} mJy/beam".format(1000 * std))
    print("Noise = {} mJy/beam".format(1000 * noise))

    x_bound = np.abs(mapsize[0] * pix_size[0] / 57.3 / 2)
    y_bound = np.abs(mapsize[1] * pix_size[1] / 57.3 / 2)
    pix_to_mas_x = 2 / mapsize[0] * x_bound * 206265000
    pix_to_mas_y = 2 / mapsize[1] * y_bound * 206265000

    # locate ridgeline
    core = np.argmax(image_data)
    x_max, y_max = np.unravel_index(core, image_data.shape)
    lmapsize = round(np.hypot(mapsize[0], mapsize[1]))
    lmap = [[[], [], []] for _ in range(lmapsize)]
    for x in np.arange(mapsize[0]):
        for y in np.arange(mapsize[1]):
            if image_data_for_ridge[x, y] > 20 * std:
                length = round(np.hypot(x - x_max, y - y_max))
                r = np.hypot((x - x_max) * pix_to_mas_x,
                             (-y + y_max) * pix_to_mas_y)
                if -y + y_max >= 0 and r > 0:
                    lmap[length][0].append(r)
                    lmap[length][1].append(np.arcsin((x - x_max) * pix_to_mas_x / r))
                    lmap[length][2].append(image_data_for_ridge[x, y])
                elif r > 0:
                    lmap[length][0].append(r)
                    lmap[length][1].append(np.pi - np.arcsin((x - x_max) * pix_to_mas_x / r))
                    lmap[length][2].append(image_data_for_ridge[x, y])

    ridgeline_polar = [[], [], []]
    ridgeline_polar[0].append(0)
    ridgeline_polar[1].append(0)
    ridgeline_polar[2].append(100)
    core_radius = round(get_appr_core_radius(image_data_for_ridge, mapsize, noise)) \
                  * np.sqrt(pix_to_mas_x * pix_to_mas_y)
    print("Estimated core radius =", core_radius, "mas")
    weight = 15
    for length_arr in lmap:
        if len(length_arr[0]) > 0:
            if length_arr[0][0] > 0:
                ridgeline_polar[0].append(np.mean(np.array(length_arr[0])))
                ridgeline_polar[1].append(circular_mean(length_arr[1], w=normalize(length_arr[2])))
                ridgeline_polar[2].append(1)
    ridgeline_polar = np.array(ridgeline_polar)
    ridgeline_polar_raw = np.copy(ridgeline_polar)

    # shift angles on 2pi
    ridgeline_polar[ridgeline_polar < 0] += 2 * np.pi
    mean = circmean(ridgeline_polar[1])
    std = circstd(ridgeline_polar[1])
    for i in np.arange(ridgeline_polar[1].size):
        if np.abs(ridgeline_polar[1][i] + 2 * np.pi - mean) < np.abs(ridgeline_polar[1][i] - mean):
            ridgeline_polar[1][i] += 2 * np.pi
        if np.abs(ridgeline_polar[1][i] - 2 * np.pi - mean) < np.abs(ridgeline_polar[1][i] - mean):
            ridgeline_polar[1][i] -= 2 * np.pi
    '''
    # delete too different points
    for i in np.arange(ridgeline_polar[2].size):
        if np.abs(ridgeline_polar[1][i] - mean) > np.pi / 3:
            ridgeline_polar[0][i] = 0

    # get points around the core with mean angle
    for i in np.arange(ridgeline[2].size-2, -1, -1):
        if ridgeline_polar[0][i] == 0:
            ridgeline_polar[1][i] = ridgeline_polar[1][i+1]
        elif ridgeline_polar[2][i] == 0:
            ridgeline_polar[1][i] = circmean(ridgeline_polar[1, i:i+5])
            ridgeline_polar[2][i] = 1
    '''
    # delete too different points
    # get points around the core with mean angle
    for i in np.arange(ridgeline_polar[2].size - 2, -1, -1):
        if ridgeline_polar[0][i] == 0:
            ridgeline_polar[1][i] = ridgeline_polar[1][i + 1]
        if np.abs(ridgeline_polar[1][i] - mean) > std:
            ridgeline_polar[1][i] = circmean(ridgeline_polar[1, i+1:i + 5])
        if ridgeline_polar[0][i] < 1.25 * core_radius:
            ridgeline_polar[1][i] = circmean(ridgeline_polar[1, i+1:i + 5])

    for i in np.arange(ridgeline_polar[1].size):
        if np.abs(ridgeline_polar[1][i] + 2 * np.pi - mean) < np.abs(ridgeline_polar[1][i] - mean):
            ridgeline_polar[1][i] += 2 * np.pi
        if np.abs(ridgeline_polar[1][i] - 2 * np.pi - mean) < np.abs(ridgeline_polar[1][i] - mean):
            ridgeline_polar[1][i] -= 2 * np.pi
        if np.isnan(ridgeline_polar[1][i]):
            ridgeline_polar[1][i] = ridgeline_polar[1][i - 1]

    ridgeline_polar = ridgeline_polar[:, ridgeline_polar[0].argsort()]
    # spl = get_univariate_spl(x=ridgeline_polar[0], y=ridgeline_polar[1], w=ridgeline_polar[2])
    while ridgeline_polar[1].max() - ridgeline_polar[1].min() > np.pi/4:

        ridgeline_polar = ridgeline_polar[:, :-1]
    spl = get_smooth_univariate_spl(x=ridgeline_polar[0], y=ridgeline_polar[1], w=ridgeline_polar[2])

    return spl, ridgeline_polar, ridgeline_polar_raw
