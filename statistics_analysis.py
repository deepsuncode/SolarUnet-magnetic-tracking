# =========================================================================
#   (c) Copyright 2020
#   All rights reserved
#   Programs written by Haodi Jiang
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    #print('turning logging of is not available')
    pass
from scipy.ndimage.measurements import label
from magnetic_tracking import size_filter, create_elements_flux_dict
from matplotlib.patches import Rectangle
from scipy.stats import epps_singleton_2samp,describe
import csv
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
from astropy.io import fits


def flux_to_MX(flux):
    c = (0.083 * 725 * 10**5) ** 2
    return abs(c * flux)


def pixel_to_Mm(size):
    return size*0.083*0.725


def read_feature_lifetime_list_file(filename):
    """
    read feature lifetime list from .csv file
    :param lst:
    :return:
    """
    with open(filename) as myFile:
        reader = csv.reader(myFile)
        data = next(reader)
        return [int(i) for i in data]


def statistics_analysis_area_flux():
    """
    this function analysis the statistics of features in a frame:
    area, flux
    """
    # output path
    outpath = 'results/statistics_analysis/'

    '''read SolarNet data'''
    fits_file = 'data/statistics_analysis/size_flux/bz20170713_201549.fits'
    mask_2_class = 'data/statistics_analysis/size_flux/SolarUnet_2_class_mask.png'

    fits_file = fits.open(fits_file)
    fits_data = fits_file[0].data
    fits_data = np.flipud(fits_data)
    mask_2 = cv2.imread(mask_2_class, 0)
    # mask_2 = cv2.bitwise_not(mask_2)  # # have to covert 255->0, 0->255, otherwise cannot find each segment

    magnetic_field_size_threshold = 2 # plot with different pixel size
    '''pre-calculate the flux and contour for all the frames'''
    solarNet_structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter
    solarNet_labeled, solarNet_ncomponents = label(mask_2, solarNet_structure)
    solarNet_indices = np.indices(mask_2.shape).T[:, :, [1, 0]]
    solarNet_frame_filtered_comp = size_filter(solarNet_labeled, solarNet_indices, solarNet_ncomponents,
                                                   magnetic_field_size_threshold)
    solarNet_frame_flux = create_elements_flux_dict(solarNet_frame_filtered_comp, fits_data)

    '''read SWAMIS data'''
    swamis_flux_size = sio.readsav('data/statistics_analysis/size_flux/SWAMIS_flux-sz.sav') # SWAMIS_flux: frame 1 data
    value_flux = swamis_flux_size['flux']
    value_sz = swamis_flux_size['sz']

    non_zero_flux = value_flux[np.nonzero(value_flux)]
    non_zero_size = value_sz[np.nonzero(value_sz)]

    ''' plot feature by pixel size '''
    non_zero_size = sorted(non_zero_size * (0.083*0.725)**2)[:-1]
    SolarNet_frame_comp_dict = solarNet_frame_filtered_comp
    SolarNet_feature_size_list = [(len(feature)) * (0.083*0.725)**2 for feature in SolarNet_frame_comp_dict.values()]
    SolarNet_feature_size_list = sorted(SolarNet_feature_size_list)

    # histogram line color
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # control legend
    handles = [Rectangle((0, 0), 1, 1, color=c, fill=False, alpha=1) for c in [color[0], color[1]]]  # alpha = 0.8 is best)
    labels = ["SWAMIS", "SolarUnet"]

    fig, axs = plt.subplots(2, sharex=True, sharey=True, gridspec_kw={'hspace': 0}, figsize=(7.25, 3))
    # plt.figure(figsize=(7.25, 2.75))
    axs[0].hist(non_zero_size, bins=100, histtype='step', stacked=True, fill=False, label='SWAMIS',
                log=True, color=color[0])
    axs[1].hist(SolarNet_feature_size_list, bins=100, histtype='step', stacked=True, fill=False, label='SolarUnet',
                log=True, color=color[1])

    axs[0].legend(handles, labels, loc='upper right', fontsize=8)

    plt.xlabel(r'Feature size (Mm$^{2}$)', fontsize=10)
    # plt.ylabel('Number of features in lifetime bin',fontsize=10)
    fig.text(0.05, 0.5, 'Number of features in size bin', fontsize=10, va='center', rotation='vertical')

    axs[0].tick_params(axis='y', labelsize=8)
    axs[1].tick_params(axis='x', labelsize=8)
    axs[1].tick_params(axis='y', labelsize=8)

    plt.savefig(outpath+'feature-size.png', bbox_inches='tight', dpi=500)
    plt.show()

    '''plot feature by flux'''
    non_zero_flux = abs(non_zero_flux)
    non_zero_flux = [feature / (10 ** 18) for feature in non_zero_flux]
    non_zero_flux = sorted(non_zero_flux)[:]

    SolarNet_frame_flux_dict = solarNet_frame_flux
    SolarNet_feature_flux_list= list(SolarNet_frame_flux_dict.values())
    SolarNetfeature_flux_list = [flux_to_MX(flux) / (10 ** 18) for flux in SolarNet_feature_flux_list]
    SolarNetfeature_flux_list = sorted(SolarNetfeature_flux_list)[:]

    '''plotting'''
    fig, axs = plt.subplots(2, sharex=True, sharey=True, gridspec_kw={'hspace': 0}, figsize=(7.25, 3))
    axs[0].hist(non_zero_flux, bins=100, histtype='step', stacked=True, fill=False, label='SWAMIS',
                log=True, color=color[0])
    axs[1].hist(SolarNetfeature_flux_list, bins=100, histtype='step', stacked=True, fill=False, label='SolarUnet',
                log=True, color=color[1])

    axs[0].legend(handles, labels, loc='upper right', fontsize=8)

    plt.xlabel(r'Magnetic flux (${10^{18}}$Mx)', fontsize=10)
    # plt.ylabel('Number of features in lifetime bin',fontsize=10)
    fig.text(0.05, 0.5, 'Number of features in flux bin', fontsize=10, va='center', rotation='vertical')

    axs[0].tick_params(axis='y', labelsize=8)
    axs[1].tick_params(axis='x', labelsize=8)
    axs[1].tick_params(axis='y', labelsize=8)

    plt.savefig(outpath+'feature-flux.png', bbox_inches='tight', dpi=500)
    plt.show()


def statistics_analysis_lifetime():
    """
    this function analysis the statistics of features in a frame:lifetime
    """
    # print('====================Plot feature lifetime distribution=======================')

    outpath = 'results/statistics_analysis/'

    solarNet_lifetime_list = read_feature_lifetime_list_file(
        'data/statistics_analysis/lifetime/feature_lifetime_SolarUnet.csv')
    solarNet_lifetime_list = sorted(solarNet_lifetime_list)

    swamis_lifetime = sio.readsav('data/statistics_analysis/lifetime/feature_lifetime_SWAMIS.sav')
    value_lt = swamis_lifetime['nm_lt']

    swamis_lifetime_list = []
    for i in range(1, len(value_lt)):
        swamis_lifetime_list += [i] * int(value_lt[i])

    color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(7.25, 3))
    plt.hist(swamis_lifetime_list, bins=range(147), histtype='step', stacked=True, fill=False, label='SWAMIS',
             color=color[0])
    plt.hist(solarNet_lifetime_list, bins=range(147), histtype='step', stacked=True, fill=False, label='SolarUnet',
             log=True, color=color[1])
    plt.legend(loc='upper right', fontsize=8)
    plt.xlabel('Lifetime (frames)', fontsize=10)
    plt.ylabel('Number of features in lifetime bin', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig(outpath + 'feature-lifetime.png', bbox_inches='tight', dpi=500)
    plt.show()


def analysis():
    statistics_analysis_area_flux()
    statistics_analysis_lifetime()

