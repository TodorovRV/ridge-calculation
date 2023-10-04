from build_ridgeline import get_ridgeline, subtract_gaussian_core, get_average_ridgeline
from astropy.io import fits
import numpy as np
import os
import scipy
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0, '/home/rtodorov/jetpol/ve/vlbi_errors')
from utils import find_bbox, find_image_std, mas_to_rad, degree_to_rad
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from image import plot as iplot
from scipy.interpolate import UnivariateSpline, interp1d


def get_p_img_for_plot(image_data_i, image_data_p, min_abs_level):
    image_data_p_for_plot = np.zeros(image_data_p.shape)
    image_data_p_for_plot[image_data_p > 3.1 * min_abs_level] = image_data_p[
        image_data_p > 3.1 * min_abs_level]
    mask = image_data_p <= 3.1 * min_abs_level
    mask[image_data_i < 3.1 * min_abs_level] = 0
    image_data_p_for_plot[mask] = 3.1 / 2 * min_abs_level
    image_data_p_for_plot[image_data_p_for_plot == 0] = 0

    image_data_p_for_plot = scipy.ndimage.gaussian_filter(image_data_p_for_plot, sigma=0.35)
    image_data_p_for_plot = scipy.ndimage.gaussian_filter(image_data_p_for_plot, sigma=0.35)
    image_data_p_for_plot = scipy.ndimage.gaussian_filter(image_data_p_for_plot, sigma=0.35)
    image_data_p_for_plot = scipy.ndimage.gaussian_filter(image_data_p_for_plot, sigma=0.35)
    return image_data_p_for_plot


def plot_sourse_map_w_ridge(source_name, data_folder, base_dir='./', outfile='./fig.jpg', contours_mode='P', 
                            colors_mode='n', vectors_mode='chi', fig=None, ax=None):
    i_file = '{}/{}_stack_i_true.fits'.format(data_folder, source_name)
    p_file = '{}/{}_stack_p_true.fits'.format(data_folder, source_name)
    chi_file = '{}/{}_stack_p_ang.fits'.format(data_folder, source_name)
    try:
        std_EVPA_file = '{}/std_evpa_fits/{}_std_EVPA_deg_unbiased_var_2.fits'.format(data_folder, source_name)
        image_data_std_EVPA = fits.getdata(std_EVPA_file, ext=0)
    except:
        print("STD EVPA is not defined!")
        image_data_std_EVPA = None
    
    image_data_i = fits.getdata(i_file, ext=0)[0][0]
    image_data_p = fits.getdata(p_file, ext=0)[0][0]
    image_data_chi = fits.getdata(chi_file, ext=0)[0][0] * degree_to_rad

    # check file sanity
    if not (image_data_i.shape == image_data_p.shape == image_data_chi.shape):
        raise Exception('Files shape do not match')

    image_data_fpol = image_data_p / image_data_i

    hdul = fits.open(i_file)
    beam = [hdul[0].header['BMAJ'] * 3600000, hdul[0].header['BMIN'] * 3600000, hdul[0].header['BPA']]
    mapsize = [hdul[0].header['NAXIS1'], hdul[0].header['NAXIS2']]
    min_abs_level = 0.5 * np.abs(hdul[0].header['DATAMIN'])
    noise = hdul[0].header['NOISE']
    pix_size = [hdul[0].header['CDELT1'], hdul[0].header['CDELT2']]
    # print(beam)
    hdul.info()

    # fix nan in img
    image_data_i[np.isnan(image_data_i)] = 0
    image_data_p[np.isnan(image_data_p)] = 0
    image_data_fpol[np.isnan(image_data_fpol)] = 0
    image_data_chi[np.isnan(image_data_chi)] = 0

    image_data_fpol[image_data_fpol > 1] = 1
    image_data_fpol[image_data_fpol < 0] = 0
    '''
    image_data_i = np.flip(image_data_i)
    image_data_fpol = np.flip(image_data_fpol)
    image_data_chi = np.flip(image_data_chi)
    '''
    core = np.argmax(image_data_i)
    x_max, y_max = np.unravel_index(core, image_data_i.shape)

    # shift img to locate core in centre
    image_data_i = np.roll(image_data_i, round(mapsize[0] / 2 - x_max), axis=0)
    image_data_i = np.roll(image_data_i, round(mapsize[1] / 2 - y_max), axis=1)
    image_data_p = np.roll(image_data_p, round(mapsize[0] / 2 - x_max), axis=0)
    image_data_p = np.roll(image_data_p, round(mapsize[1] / 2 - y_max), axis=1)
    image_data_fpol = np.roll(image_data_fpol, round(mapsize[0] / 2 - x_max), axis=0)
    image_data_fpol = np.roll(image_data_fpol, round(mapsize[1] / 2 - y_max), axis=1)
    image_data_chi = np.roll(image_data_chi, round(mapsize[0] / 2 - x_max), axis=0)
    image_data_chi = np.roll(image_data_chi, round(mapsize[1] / 2 - y_max), axis=1)

    npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
    std = find_image_std(image_data_i, beam_npixels=npixels_beam)
    std_p = find_image_std(image_data_p, beam_npixels=npixels_beam)
    min_abs_level = 2 * std
    blc, trc = find_bbox(image_data_i, level=20 * std, min_maxintensity_mjyperbeam=40 * std,
                         min_area_pix=4 * npixels_beam, delta=10)
    if blc[0] == 0: blc = (blc[0] + 1, blc[1])
    if blc[1] == 0: blc = (blc[0], blc[1] + 1)
    if trc[0] == image_data_i.shape: trc = (trc[0] - 1, trc[1])
    if trc[1] == image_data_i.shape: trc = (trc[0], trc[1] - 1)
    # blc = [400, 400]
    # trc = [624, 624]

    x_bound = np.abs(mapsize[0] * pix_size[0] / 57.3 / 2)
    y_bound = np.abs(mapsize[1] * pix_size[1] / 57.3 / 2)

    # plot_ridgeline(image_data, beam, mapsize, min_abs_level, std, pix_size)
    spl, ridge, raw_ridge = get_average_ridgeline(image_data_i, beam, mapsize, min_abs_level, std, pix_size)
    if spl is None:
        print('!!! Unable to build ridgeline !!!')
        return None

    label_size = 16
    plt.rcParams['xtick.labelsize'] = label_size
    plt.rcParams['ytick.labelsize'] = label_size
    plt.rcParams['axes.titlesize'] = label_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['font.size'] = label_size
    plt.rcParams['legend.fontsize'] = label_size
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    if fig is None:
        fig = plt.figure(figsize=(8.5, 6))
    if ax is None:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r'Relative R.A. (mas)')
    ax.set_ylabel(r'Relative Decl. (mas)')
    # fig.set_size_inches(4.5, 3.5)

    # maxlen_coord = ridge[0].flat[abs(ridge[1]).argmax()]
    maxlen_coord = np.max(ridge[0])
    minlen_coord = np.min(ridge[ridge > 0])
    ridge_x = ridge[0] * np.sin(ridge[1])
    ridge_y = ridge[0] * np.cos(ridge[1])

    rs = np.linspace(0, maxlen_coord, 1000)
    thetas = spl(rs)
    # plt.plot(rs, thetas)
    # plt.show()
    if True:
        ax.plot(rs * np.cos(thetas), rs * np.sin(thetas))
        # ax.scatter(ridge_y, ridge_x)

        if contours_mode == 'I':
            contours = image_data_i
            P_colorscheme = False
        elif contours_mode == 'P':
            contours = get_p_img_for_plot(image_data_i, image_data_p, min_abs_level)
            P_colorscheme = True
        elif contours_mode == 'fpol':
            contours = image_data_fpol
            P_colorscheme = False
        else:
            contours = None

        if colors_mode == 'I':
            colors = image_data_i
            plot_colorbar = True
            colorbar_label = 'Flux, mJa/beam'
            colors_mask = [image_data_i < 20 * std]
        elif colors_mode == 'P':
            colors = get_p_img_for_plot(image_data_i, image_data_p, min_abs_level)
            plot_colorbar = True
            colorbar_label = 'Polarized flux, mJa/beam'
            colors_mask = [image_data_i < 20 * std]
        elif colors_mode == 'fpol':
            colors = image_data_fpol
            plot_colorbar = True
            colorbar_label = 'Frac. pol.'
            colors_mask = [image_data_i < 20 * std]
        elif colors_mode == 'std':
            colors = image_data_std_EVPA
            plot_colorbar = True
            colorbar_label = 'STD EVPA, deg'
            colors_mask = [image_data_i < 20 * std]
        else:
            colors = None
            plot_colorbar = False
            colorbar_label = None
            colors_mask = [image_data_i < 20 * std]

        if vectors_mode == 'chi':
            vectors = image_data_chi
            vectors_mask = [image_data_i < 20 * std]
        else:
            vectors = None
            vectors_mask = None

        iplot(contours=contours,  # subtract_gaussian_core(image_data_i, mapsize, 40*std),
              colors=colors, colors_mask=colors_mask,
              vectors=vectors, vectors_mask=vectors_mask,
              x=np.linspace(-x_bound, x_bound, mapsize[1]), show_beam=True, k=2, vinc=4, cmap='Oranges',
              y=np.linspace(y_bound, -y_bound, mapsize[0]), min_abs_level=2 * min_abs_level,
              blc=blc, trc=trc, close=False, contour_color='black', plot_colorbar=plot_colorbar,
              beam=beam, fig=fig, axes=ax, label_size=label_size, colorbar_label=colorbar_label,
              P_colorcheme=P_colorscheme)
        # plt.show()
        # fig.savefig('./fig.jpg')
        fig.savefig(os.path.join(base_dir+'/source_graphs', outfile))


if __name__ == "__main__":
    data_folder = '/home/rtodorov/data_stack_fits'
    base_dir = '/home/rtodorov/ridge-calculation'
    
    with open(os.path.join(data_folder, "source_list.txt")) as f:
        lines = f.readlines()

    for line in lines:
        source_name = line[0:8]
        plot_sourse_map_w_ridge(source_name, data_folder, base_dir=base_dir, outfile='{}.jpg'.format(source_name), 
                                contours_mode='I', colors_mode='n', vectors_mode='n')
