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

import numpy as np
from astropy.io import fits
import cv2
from scipy.ndimage.measurements import label
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import itertools
import operator
import time
import os
import csv


def read_data_files(input_path):
    fits_files_dict = {}
    image_dict = {}
    mask_2_class_dict = {}
    mask_3_class_dict = {}

    path_fits = input_path + 'fits/'
    path_image = input_path + 'image/'
    path_mask_2_class = input_path + 'mask_2_class/'
    path_mask_3_class = input_path + 'mask_3_class/'

    num_frames = len(os.listdir(path_fits))
    fits_key = 1
    for name in os.listdir(path_fits):
        # read the corresponding fits data
        fits_file = fits.open(path_fits + name)
        fits_data = fits_file[0].data
        fits_data = np.flipud(fits_data)
        fits_files_dict[fits_key] = fits_data
        fits_key += 1

    image_key = 1
    for name in os.listdir(path_image):
        image = cv2.imread(path_image + name)
        image_dict[image_key] = image
        image_key += 1

    mask_2_class_key = 1
    for name in os.listdir(path_mask_2_class):
        mask_2 = cv2.imread(path_mask_2_class + name, 0)
        mask_2 = cv2.bitwise_not(mask_2)   # # have to covert 255->0, 0->255, otherwise cannot find each segment
        mask_2_class_dict[mask_2_class_key] = mask_2
        mask_2_class_key += 1

    mask_3_class_key = 1
    for name in os.listdir(path_mask_3_class):
        mask_3 = cv2.imread(path_mask_3_class + name, 0)
        mask_3_class_dict[mask_3_class_key] = mask_3
        mask_3_class_key += 1

    return fits_files_dict, image_dict, mask_2_class_dict, mask_3_class_dict, num_frames


def pre_calulate_flux_and_contour(fits_files_dict, mask_2_class_dict, num_frames, magnetic_field_size_threshold=10):
    '''
    pre-calculate the flux and contour for all the frames
    magnetic_field_size_threshold = 10  # minimum size for element to be considered
    '''
    filtered_comp_dict = {}
    flux_dict = {}
    contour_dict = {}

    for i in range(1, num_frames + 1):
        cur_fits = fits_files_dict[i]
        cur_mask_2_class = mask_2_class_dict[i]

        structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter
        labeled, ncomponents = label(cur_mask_2_class, structure)
        indices = np.indices(cur_mask_2_class.shape).T[:, :, [1, 0]]

        frame_filtered_comp = size_filter(labeled, indices, ncomponents, magnetic_field_size_threshold)
        # print('frame {} filtered_comp size'.format(i), len(frame_filtered_comp))
        frame_flux_dict = create_elements_flux_dict(frame_filtered_comp, cur_fits)
        # print('----------------frame {} flux calculation done----------------'.format(i))
        frame_contour_dict = create_elements_contour_dict(frame_filtered_comp, cur_mask_2_class)
        # print('----------------frame {} contour calculation done----------------'.format(i))
        filtered_comp_dict[i] = frame_filtered_comp
        flux_dict[i] = frame_flux_dict
        contour_dict[i] = frame_contour_dict
    return filtered_comp_dict, flux_dict, contour_dict


# filter the size of predicted_mask element
def size_filter(labeled, indices, n, size):
    new_element_dict = {}
    label_count = 1
    for i in range(1, n+1):
        if len(indices[labeled == i]) >= size:
            # print('label count: ',label_count)
            new_element_dict[label_count] = indices[labeled == i]
            label_count += 1
    return new_element_dict


# filter the size of predicted_mask element
def size_filter_v1(labeled, indices, n, min_size, max_size):
    new_element_dict = {}
    label_count = 1
    for i in range(1, n+1):
        if len(indices[labeled == i]) >= min_size and len(indices[labeled == i]) <= max_size:
            new_element_dict[label_count] = indices[labeled == i]
            label_count += 1
    return new_element_dict


# use the average value of the pixels to define the white or black
def element_pos_neg(component, img_3_class):
    sum = 0
    for comp in component:
        sum += img_3_class[comp[0]][comp[1]]
    average = sum / len(component)
    if average >= 128:
        return True
    else:
        return False

# find the out whether predicted_mask element in frame 1 and frame 2 with same sign
def check_same_pos_neg(filtered_element_1, filtered_element_2, img_3_class_1, img_3_class_2):

    sum_1= 0
    for comp in filtered_element_1:
        sum_1 += img_3_class_1[comp[0]][comp[1]]
    average_1 = sum_1 / len(filtered_element_1)
    if average_1 >= 128:
        value_1 = 255
    else:
        value_1 = 0

    sum_2= 0
    for comp in filtered_element_2:
        sum_2 += img_3_class_2[comp[0]][comp[1]]
    average_2 = sum_2 / len(filtered_element_2)
    if average_2 >= 128:
        value_2 = 255
    else:
        value_2 = 0

    if value_1 == value_2:
        return True
    else:
        return False


# use the threshold to decide whether two predicted_mask element are the same
def check_with_threshold(filtered_element_1, filtered_element_2, threshold):
    mag_element_num_1 = len(filtered_element_1)
    mag_element_num_2 = len(filtered_element_2)
    common_element = set((map(tuple, filtered_element_1))).intersection(set(map(tuple, filtered_element_2)))
    if len(common_element) / mag_element_num_1 >= threshold:
        return True
    else:
        return False


# check difference between two elements threshold'''
def check_with_threshold_2(element_1, element_2, threshold2):
    '''
    :param element_1: target element
    :param element_2: element in the other image to compare with
    :param threshold: given by user, 0.6
    :return: True or False
    '''
    mag_element_num_1 = len(element_1)
    mag_element_num_2 = len(element_2)
    if abs((mag_element_num_1 - mag_element_num_2) / mag_element_num_1) <= threshold2:
        return True
    else:
        return False


#  returns the vertices as a set of nested lists of contour lines, contour sections and arrays of x,y vertices
# find the contour line of a component
def find_contour_line(component, mask_2_class):
    contour_line = []
    component = component.tolist()
    height, width = mask_2_class.shape
    for pixel in component:
        if pixel[0] == 0 or pixel[0] == height-1 or pixel[1] == 0 or pixel[1] == width-1:
            contour_line.append(pixel)
        else:
            p_left = mask_2_class[pixel[0], pixel[1]-1]
            p_right = mask_2_class[pixel[0], pixel[1]+1]
            p_up = mask_2_class[pixel[0]-1, pixel[1]]
            p_down = mask_2_class[pixel[0]+1, pixel[1]]

            p_fits_value_list = [p_left, p_right, p_up, p_down]
            if 0 in p_fits_value_list: #or backgroud_sign in p_fits_value_list:
                contour_line.append(pixel)
    contour_line = np.asarray(contour_line)

    return contour_line


# results draw and find contour line
def find_and_draw_contour_line_new(img,contour_line, img_3_class):
    green = [0, 255, 0]
    yellow = [0, 255, 255]
    for p in contour_line:
        p_value = img_3_class[p[0]][p[1]]
        if p_value == 255:
            img[p[0], p[1]] = yellow
        else:
            img[p[0], p[1]] = green


def draw_contour_line_event(img, contour_line, color):
    for p in contour_line:
        img[p[0], p[1]] = color  # red


def draw_contour_line_event_half(img, contour_line, color):
    for i in range(len(contour_line)//2):
        p = contour_line[i]
        img[p[0], p[1]] = color  # red


def put_num_on_element(img, component, label_number):

    center = np.round(component.mean(0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_color = [0, 0, 255]
    label_color = [0, 195, 255]
    if len(component) > 1000:
        label_size = 0.4
        # color = [0, 0 , 255] # red
    elif len(component) > 500:
        label_size = 0.3
        # color = [0, 0 , 255] # amber
    elif len(component) > 100:
        label_size = 0.3
        # color = [0, 0 , 255] # amber
    else:
        label_size = 0.275
        # color = [0, 0, 255]
    cv2.putText(img, str(label_number), (int(center[1]), int(center[0])), font, label_size, label_color, 1, cv2.LINE_AA)


def find_the_radius(center, points_contour):
    first_point = points_contour[0]
    last_point = points_contour[-1]

    dis_first = distance.euclidean(center, first_point)
    dis_last = distance.euclidean(center, last_point)
    if dis_first >= dis_last:
        radius = dis_first
    else:
        radius = dis_last
    return np.ceil(radius)


def element_moveing_distance(cadence, pixel_resolution=0.083):
    '''cadence: time gap between each frame
       pixel_resolution: from the telescope observation and
                        in the paper we choose 0.083"/pixel
    '''
    distance  = (4*cadence) / (725*pixel_resolution)
    return np.ceil(distance)


def RoI(img, center, points_contour, cadence ,pixel_resolution= 0.083):
    height, width, c = img.shape
    radius = 0
    for p in points_contour:
        dis_temp = distance.euclidean(center, p)
        if dis_temp >= radius:
            radius = dis_temp

    radius = np.ceil(radius)
    moving_distance = np.ceil((4*cadence) / (725*pixel_resolution))

    roi = 1 * (2 * moving_distance + 1 * radius)
    x_range_min = center[0] - roi
    x_range_max = center[0] + roi
    if x_range_min < 0:
        x_range_min = 0
    if x_range_max > height:
        x_range_max = height

    y_range_min = center[1] - roi
    y_range_max = center[1] + roi

    if y_range_min < 0:
        y_range_min = 0

    if y_range_max > width:
        y_range_max = width

    return ((x_range_min, x_range_max), (y_range_min, y_range_max))


# '''3-2-2019 add a element region range'''
def element_region(img, center, points_contour):
    height, width, c = img.shape
    first_point = points_contour[0]
    last_point = points_contour[-1]
    dis_first = distance.euclidean(center, first_point)
    dis_last = distance.euclidean(center, last_point)
    if dis_first >= dis_last:
        radius = dis_first
    else:
        radius = dis_last

    radius = np.ceil(radius)
    roi = 1 * radius # defaut 1
    x_range_min = center[0] - roi
    x_range_max = center[0] + roi
    if x_range_min < 0:
        x_range_min = 0
    if x_range_max > height:
        x_range_max = height

    y_range_min = center[1] - roi
    y_range_max = center[1] + roi

    if y_range_min < 0:
        y_range_min = 0

    if y_range_max > width:
        y_range_max = width

    return ((x_range_min, x_range_max), (y_range_min, y_range_max))


# make a dict for the contour line of element with label
def create_elements_contour_dict(filtered_img_elements, mask_2_class):
    dict_elements_contour = {}  # key: label 1,2,3  ; value: element contour
    for i in range(1, len(filtered_img_elements) + 1):
        element_contour = find_contour_line(filtered_img_elements[i], mask_2_class)
        dict_elements_contour[i] = element_contour
    return dict_elements_contour


# make a dict for the predicted_mask flux of element with label
def create_elements_flux_dict(filtered_img_elements, fits_data):
    dict_elements_flux = {}  # key: label 1,2,3  ; value: element contour
    for i in range(1, len(filtered_img_elements) + 1):
        element_component = filtered_img_elements[i]
        flux = 0
        for p in element_component:
            flux += fits_data[p[0], p[1]]
        dict_elements_flux[i] = flux
    return dict_elements_flux


def create_elements_flux_size_dict(filtered_img_elements, fits_data):
    dict_elements_flux = {}  # key: label 1,2,3  ; value: element contour
    for i in range(1, len(filtered_img_elements) + 1):
        element_component = filtered_img_elements[i]
        size = len(element_component)
        flux = 0
        for p in element_component:
            flux += fits_data[p[0], p[1]]
        dict_elements_flux[i] = (flux, size)
    return dict_elements_flux


# calculate element average distance between two predicted_mask elmenet
def calculate_element_average_distance(e1, e2):
    element_distance = 0
    for p1 in e1:
        for p2 in e2:
            element_distance += distance.euclidean(p1, p2)
    return element_distance/(len(e1)*len(e2))


# minimum distance between two predicted_mask flux elements
def calculate_element_minimum_distance_old(e1, e2):
    # points_pairwise_distance_list = []
    min_distance = 9999999999
    for p1 in e1:
        for p2 in e2:
            d = distance.euclidean(p1, p2)
            if d < min_distance:
                min_distance = d
            # points_pairwise_distance_list.append(distance.euclidean(p1, p2))
    # return min(points_pairwise_distance_list)
    return min_distance


# use KD tree to find minimum distance
def calculate_element_minimum_distance(e1, e2):
    if e1.size <= e2.size:
        target_x = e1
        obj_x = e2
    else:
        target_x = e2
        obj_x = e1

    tree = KDTree(obj_x)
    distance_list_tar_obj = []
    for point in target_x:
        p = np.array([point]) # search one.
        nearest_dist, nearest_ind = tree.query(p, k=1)  # k=2 nearest neighbors where k1 = identity
        distance_list_tar_obj.append(nearest_dist[0][0])
    min_dis = min(distance_list_tar_obj)
    return min_dis


def find_nearest_n_neighbor_contour(neighbor_group, object_element_contour, contour_dict, nearest_num):
    element_distance_dict = {}
    for e_label in neighbor_group:
        e = contour_dict[e_label]
        # element_distance_dict[e_label] = calculate_element_average_distance(object_element, e)   # commented 9-6
        element_distance_dict[e_label] = calculate_element_minimum_distance(object_element_contour, e)
    nearest_n = dict(sorted(element_distance_dict.items(), key=operator.itemgetter(1))[:nearest_num])
    nearest_n_same = list(nearest_n.keys())
    return nearest_n_same


def find_nearest_n_neighbor(neighbor_group, object_element, filtered_img, nearest_num):
    element_distance_dict = {}
    for e_label in neighbor_group:
        e = filtered_img[e_label]
        # element_distance_dict[e_label] = calculate_element_average_distance(object_element, e)
        element_distance_dict[e_label] = calculate_element_minimum_distance(object_element, e)
    nearest_n = dict(sorted(element_distance_dict.items(), key=operator.itemgetter(1))[:nearest_num])
    nearest_n_same = list(nearest_n.keys())
    return nearest_n_same


def event_tracking_forward(i, cur_original_image, cur_filtered_comp, next_filtered_comp, cur_flux_dict, next_flux_dict,
            cur_contour_dict, next_contour_dict, cur_mask_3, cadence, pixel_resolution, cur_roi,
                        object_element_roi_next_image, object_element_flux_1, img_comp_cur, points_contour, label_option): # , flag_first=False):

    roi = cur_roi
    roi_x_range = roi[0]
    roi_y_range = roi[1]
    object_neighbor_element_roi_cur_image_same = []
    object_neighbor_element_roi_cur_image_dif = []
    for jj in range(1, len(cur_filtered_comp) + 1):
        if jj != i:
            element_contour_points = cur_contour_dict[jj]
            flag = False
            for p in element_contour_points:
                if roi_x_range[0] <= p[0] <= roi_x_range[1] and roi_y_range[0] <= p[1] <= roi_y_range[1]:
                    flag = True
                    break
            if flag == True:
                # '''check this element in the roi in the next image '''
                neighbor_element_jj = cur_filtered_comp[jj]
                neighbor_element_flux_jj = cur_flux_dict[jj]
                neighbor_center = np.round(neighbor_element_jj.mean(0))
                neighbor_roi = RoI(cur_original_image, neighbor_center, element_contour_points, cadence,
                                   pixel_resolution)  # fin
                neighbor_roi_x_range = neighbor_roi[0]
                neighbor_roi_y_range = neighbor_roi[1]
                #'''work in next image'''
                key_flag_2 = False
                for k in range(1, len(next_filtered_comp) + 1):  # range should given by jj
                    element_contour_points = next_contour_dict[k]
                    flag = False
                    for p in element_contour_points:
                        if neighbor_roi_x_range[0] <= p[0] <= neighbor_roi_x_range[1] and neighbor_roi_y_range[
                            0] <= p[1] <= neighbor_roi_y_range[1]:
                            flag = True
                            break
                    if flag == True:
                        flux = next_flux_dict[k]
                        theta = abs((neighbor_element_flux_jj - flux) / neighbor_element_flux_jj)
                        if theta <= 0.1:
                            key_flag_2 = True
                            break
                if key_flag_2 == True:
                    continue
                else:
                    if check_same_pos_neg(cur_filtered_comp[i], cur_filtered_comp[jj], cur_mask_3, cur_mask_3):
                        object_neighbor_element_roi_cur_image_same.append(jj)
                    else:
                        object_neighbor_element_roi_cur_image_dif.append(jj)

    # filter the group to size 10 by find the nearest 10 element
    neighbor_number = 10 #
    if len(object_neighbor_element_roi_cur_image_same) > neighbor_number:
        object_neighbor_element_roi_cur_image_same = find_nearest_n_neighbor_contour(object_neighbor_element_roi_cur_image_same,
                                                                             cur_contour_dict[i], cur_contour_dict, neighbor_number)
    # find all the combinations of group same and dif
    group_same_combinations = []
    if len(object_neighbor_element_roi_cur_image_same) <= neighbor_number:
        for group_same_i in range(1, len(object_neighbor_element_roi_cur_image_same) + 1):
            comb = itertools.combinations(object_neighbor_element_roi_cur_image_same, group_same_i)
            group_same_combinations.append(list(comb))
    else:
        for group_same_i in range(1, neighbor_number):
            comb = itertools.combinations(object_neighbor_element_roi_cur_image_same, group_same_i)
            group_same_combinations.append(list(comb))
    # check merging mark as orange color access the comb in the group
    merging_flag = False
    for y in object_element_roi_next_image:
        y_flux = next_flux_dict[y]
        for group in group_same_combinations:
            for comb in group:
                comb_flux = 0
                for e in comb:
                    comb_flux += cur_flux_dict[e]
                # checking Merging Criterion
                merging_theta = abs((object_element_flux_1 + comb_flux - y_flux) / y_flux)
                merging_t = 0.8
                if merging_theta <= merging_t:
                    color = [0, 165, 255]  # orange
                    draw_contour_line_event(cur_original_image,points_contour, color)  # for all,
                    # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
                    for ee in comb:
                        element_contour_points = cur_contour_dict[ee]
                        draw_contour_line_event(cur_original_image, element_contour_points, color)
                        # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3) #
                    if label_option == 0:
                        pass
                    elif label_option == 1:
                        put_num_on_element(cur_original_image, img_comp_cur, i)
                    elif label_option == 2:
                        pass
                    merging_flag = True
                    break
            if merging_flag == True:
                break
        if merging_flag == True:
            break
    if merging_flag == True:
        end_time = time.time()
        # print('merging element {} time cost {}'.format(i, end_time - start_time))
    else:
        # '''check Cancellation mark as pink'''
        if len(object_neighbor_element_roi_cur_image_dif) > 10:
            object_neighbor_element_roi_cur_image_dif = find_nearest_n_neighbor_contour(
                object_neighbor_element_roi_cur_image_dif,
                cur_contour_dict[i], cur_contour_dict, 10)
        group_dif_combinations = []
        if len(object_neighbor_element_roi_cur_image_dif) <= 10:
            for group_dif_i in range(1, len(object_neighbor_element_roi_cur_image_dif) + 1):
                comb = itertools.combinations(object_neighbor_element_roi_cur_image_dif, group_dif_i)
                group_dif_combinations.append(list(comb))
        else:
            for group_dif_i in range(1, 10):
                comb = itertools.combinations(object_neighbor_element_roi_cur_image_dif, group_dif_i)
                group_dif_combinations.append(list(comb))
        # add one more condition for cancellation
        cancellation_flag_1 = False
        for group in group_dif_combinations:
            for comb in group:
                comb_flux = 0
                for e in comb:
                    comb_flux += cur_flux_dict[e]
                # checking Merging Criterion
                cancellation_theta_1 = abs((object_element_flux_1 + comb_flux) / object_element_flux_1)
                if cancellation_theta_1 <= 0.5:
                    color = [180, 105, 255]  # pink
                    draw_contour_line_event(cur_original_image,points_contour, color)
                    # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
                    for ee in comb:
                        element_contour_points = cur_contour_dict[ee]
                        draw_contour_line_event(cur_original_image, element_contour_points, color)
                    if label_option == 0:
                        pass
                    elif label_option == 1:
                        put_num_on_element(cur_original_image, img_comp_cur, i)
                    elif label_option == 2:
                        pass
                    cancellation_flag_1 = True
                    break
            if cancellation_flag_1 == True:
                break
        # condition 2 for cancellation:
        cancellation_flag_2 = False
        if cancellation_flag_1 == False:
            for y in object_element_roi_next_image:
                y_flux = next_flux_dict[y]
                for group in group_dif_combinations:
                    for comb in group:
                        comb_flux = 0
                        for e in comb:
                            comb_flux += cur_flux_dict[e]
                        # checking Merging Criterion
                        cancellation_theta_2 = abs(abs((abs(object_element_flux_1) - abs(comb_flux)) - abs(y_flux)) / abs(y_flux))
                        cancellation_t2 = 0.5
                        if cancellation_theta_2 <= cancellation_t2:
                            color = [180, 105, 255]  # pink
                            draw_contour_line_event(cur_original_image, points_contour, color)  # for all, commented 8-15
                            # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3) # for paper
                            for ee in comb:
                                element_contour_points = cur_contour_dict[ee]
                                draw_contour_line_event(cur_original_image, element_contour_points, color)
                            if label_option == 0:
                                pass
                            elif label_option == 1:
                                put_num_on_element(cur_original_image, img_comp_cur, i)
                            elif label_option == 2:
                                pass
                            cancellation_flag_2 = True
                            break
                    if cancellation_flag_2 == True:
                        break
                if cancellation_flag_2 == True:
                    break
        if cancellation_flag_1 == True or cancellation_flag_2 == True:
            end_time = time.time()
            # print('cancellation element {} time cost {}'.format(i, end_time - start_time))
        else:
            color = [128, 0, 128]  # purple  death
            draw_contour_line_event(cur_original_image,points_contour, color)
            # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
            if label_option == 0:
                pass
            elif label_option == 1:
                put_num_on_element(cur_original_image, img_comp_cur, i)
            elif label_option == 2:
                pass


def event_tracking_backward(i, cur_original_image, cur_filtered_comp, previous_filtered_comp, cur_flux_dict, previous_flux_dict,
            cur_contour_dict, previous_contour_dict, cur_mask_3, cadence, pixel_resolution, cur_roi,
                        object_element_roi_previous_image, object_element_flux_1, img_comp_cur, points_contour, label_option, flag_half=False):

    roi = cur_roi
    roi_x_range = roi[0]
    roi_y_range = roi[1]
    object_neighbor_element_roi_cur_image_same = []
    object_neighbor_element_roi_cur_image_dif = []
    for jj in range(1, len(cur_filtered_comp) + 1):
        if jj != i:
            element_contour_points = cur_contour_dict[jj]
            flag = False
            for p in element_contour_points:
                if roi_x_range[0] <= p[0] <= roi_x_range[1] and roi_y_range[0] <= p[1] <= roi_y_range[1]:
                    flag = True
                    break
            if flag == True:
                # '''check this element in the roi in the next image '''
                neighbor_element_jj = cur_filtered_comp[jj]
                neighbor_element_flux_jj = cur_flux_dict[jj]

                neighbor_center = np.round(neighbor_element_jj.mean(0))
                neighbor_roi = RoI(cur_original_image, neighbor_center, element_contour_points, cadence,
                                   pixel_resolution)  # find the neighbor roi
                neighbor_roi_x_range = neighbor_roi[0]
                neighbor_roi_y_range = neighbor_roi[1]
                # '''work in next image'''
                key_flag_2 = False
                for k in range(1, len(previous_filtered_comp) + 1):  # range should given by jj
                    element_contour_points = previous_contour_dict[k]
                    flag = False
                    for p in element_contour_points:
                        if neighbor_roi_x_range[0] <= p[0] <= neighbor_roi_x_range[1] and neighbor_roi_y_range[
                            0] <= p[1] <= neighbor_roi_y_range[1]:
                            flag = True
                            break
                    if flag == True:
                        flux = previous_flux_dict[k]
                        theta = abs((neighbor_element_flux_jj - flux) / neighbor_element_flux_jj)
                        if theta <= 0.1:
                            key_flag_2 = True
                            break
                if key_flag_2 == True:
                    continue
                else:
                    if check_same_pos_neg(cur_filtered_comp[i], cur_filtered_comp[jj], cur_mask_3, cur_mask_3):
                        object_neighbor_element_roi_cur_image_same.append(jj)
                    else:
                        object_neighbor_element_roi_cur_image_dif.append(jj)

    # '''filter the group to size 10 by find the nearest five element'''
    neighbor_number = 10
    if len(object_neighbor_element_roi_cur_image_same) > neighbor_number:
        object_neighbor_element_roi_cur_image_same = find_nearest_n_neighbor_contour(object_neighbor_element_roi_cur_image_same,
                                                                             img_comp_cur, cur_filtered_comp, neighbor_number)
    # '''find all the combinations of group same and dif'''
    group_same_combinations = []
    if len(object_neighbor_element_roi_cur_image_same) <= neighbor_number:
        for group_same_i in range(1, len(object_neighbor_element_roi_cur_image_same) + 1):
            comb = itertools.combinations(object_neighbor_element_roi_cur_image_same, group_same_i)
            group_same_combinations.append(list(comb))
    else:
        for group_same_i in range(1, neighbor_number):
            comb = itertools.combinations(object_neighbor_element_roi_cur_image_same, group_same_i)
            group_same_combinations.append(list(comb))
    # check merging mark as orange color access the comb in the group
    splitting_flag = False
    for y in object_element_roi_previous_image:
        y_flux = previous_flux_dict[y]
        for group in group_same_combinations:
            for comb in group:
                comb_flux = 0
                for e in comb:
                    comb_flux += cur_flux_dict[e]
                # checking Merging Criterion
                splitting_theta = abs((object_element_flux_1 + comb_flux - y_flux) / y_flux)
                if splitting_theta <= 0.5:
                    color = [255, 255, 0]  # Cyan / Aqua
                    if not flag_half:
                        draw_contour_line_event(cur_original_image, points_contour, color)
                        # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
                    else:
                        draw_contour_line_event_half(cur_original_image, points_contour, color)
                        # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
                    for ee in comb:
                        element_contour_points = cur_contour_dict[ee]
                        draw_contour_line_event(cur_original_image, element_contour_points, color)
                    if label_option == 0:
                        pass
                    elif label_option == 1:
                        put_num_on_element(cur_original_image, img_comp_cur, i)
                    elif label_option == 2:
                        pass
                    splitting_flag = True
                    break
            if splitting_flag == True:
                break
        if splitting_flag == True:
            break
    if splitting_flag == True:
        end_time = time.time()
        # print('splitting element {} time cost {}'.format(i, end_time - start_time))
    else:
        # '''check emerging mark as red'''
        if len(object_neighbor_element_roi_cur_image_dif) > 10:
            object_neighbor_element_roi_cur_image_dif = find_nearest_n_neighbor_contour(
                object_neighbor_element_roi_cur_image_dif,
                cur_contour_dict[i], cur_contour_dict, 10)

        group_dif_combinations = []
        if len(object_neighbor_element_roi_cur_image_dif) <= 10:
            for group_dif_i in range(1, len(object_neighbor_element_roi_cur_image_dif) + 1):
                comb = itertools.combinations(object_neighbor_element_roi_cur_image_dif, group_dif_i)
                group_dif_combinations.append(list(comb))
        else:
            for group_dif_i in range(1, 10):
                comb = itertools.combinations(object_neighbor_element_roi_cur_image_dif, group_dif_i)
                group_dif_combinations.append(list(comb))
        emerging_flag_1 = False
        for group in group_dif_combinations:
            for comb in group:
                comb_flux = 0
                for e in comb:
                    comb_flux += cur_flux_dict[e]
                # checking Merging Criterion
                emerging_theta_1 = abs((object_element_flux_1 + comb_flux) / object_element_flux_1)
                if emerging_theta_1 <= 0.5:  # 0.1
                    color = [0, 0, 255]  # now: red ; teal [128, 128, 0]
                    if not flag_half:
                        draw_contour_line_event(cur_original_image, points_contour, color)
                        # put_num_on_element(cur_original_image, img_comp_cur, i)
                        for ee in comb:
                            element_contour_points = cur_contour_dict[ee]
                            draw_contour_line_event(cur_original_image, element_contour_points, color)
                            # put_num_on_element(cur_original_image, cur_filtered_comp[ee], ee)
                    else:
                        draw_contour_line_event_half(cur_original_image, points_contour, color)
                        # put_num_on_element(cur_original_image, img_comp_cur, i)
                        for ee in comb:
                            element_contour_points = cur_contour_dict[ee]
                            draw_contour_line_event(cur_original_image, element_contour_points, color)
                            # put_num_on_element(cur_original_image, cur_filtered_comp[ee], ee)
                    if label_option == 0:
                        pass
                    elif label_option == 1:
                        put_num_on_element(cur_original_image, img_comp_cur, i)
                    elif label_option == 2:
                        pass
                    emerging_flag_1 = True
                    break
            if emerging_flag_1 == True:
                break
        #add 2nd condition for emerging
        emerging_flag_2 = False
        if emerging_flag_1 == False:
            for y in object_element_roi_previous_image:
                y_flux = previous_flux_dict[y]
                for group in group_dif_combinations:
                    for comb in group:
                        comb_flux = 0
                        for e in comb:
                            comb_flux += cur_flux_dict[e]
                        # checking emerging Criterion
                        # emerging_theta_2 = abs(abs(object_element_flux_1 + comb_flux) - abs(y_flux)) / abs(y_flux)
                        emerging_theta_2 = abs(abs(object_element_flux_1) - abs(abs(comb_flux) - abs(y_flux))) / abs(
                            object_element_flux_1)
                        if emerging_theta_2 <= 0.5:  # 0.1
                            color = [0, 0, 255]  # now: red ; teal [128, 128, 0]
                            if not flag_half:
                                draw_contour_line_event(cur_original_image, points_contour, color)
                                # put_num_on_element(cur_original_image, img_comp_cur, i)
                                for ee in comb:
                                    element_contour_points = cur_contour_dict[ee]
                                    draw_contour_line_event(cur_original_image, element_contour_points, color)
                                    # put_num_on_element(cur_original_image, cur_filtered_comp[ee], ee)
                            else:
                                draw_contour_line_event_half(cur_original_image, points_contour, color)
                                # put_num_on_element(cur_original_image, img_comp_cur, i)
                                for ee in comb:
                                    element_contour_points = cur_contour_dict[ee]
                                    draw_contour_line_event(cur_original_image, element_contour_points, color)
                                    # put_num_on_element(cur_original_image, cur_filtered_comp[ee], ee)
                            if label_option == 0:
                                pass
                            elif label_option == 1:
                                put_num_on_element(cur_original_image, img_comp_cur, i)
                            elif label_option == 2:
                                pass
                            emerging_flag_2 = True
                            break
                    if emerging_flag_2 == True:
                        break
                if emerging_flag_2 == True:
                    break

        if emerging_flag_1 == True or emerging_flag_2 == True:
            end_time = time.time()
            # print('emerging element {} time cost {}'.format(i, end_time - start_time))
        else:
            color = [255, 0, 0]  # blue  birth
            if not flag_half:
                draw_contour_line_event(cur_original_image, points_contour, color)
                # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
            else:
                draw_contour_line_event_half(cur_original_image, points_contour, color)
                # find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
            if label_option == 0:
                pass
            elif label_option == 1:
                put_num_on_element(cur_original_image, img_comp_cur, i)
            elif label_option == 2:
                pass
            end_time = time.time()
            # print('birth element {} time cost {}'.format(i, end_time - start_time))


def forward(remain_stable_feature, id, feature_dict, frame_num, cur_original_image, cur_filtered_comp, next_filtered_comp, cur_flux_dict, next_flux_dict, cur_contour_dict,
            next_contour_dict, cur_mask_3, next_mask_3, cadence, pixel_resolution, label_option):
    overlap_threshold = 0.01
    dif_threshold = 1
    for i in range(1, len(cur_filtered_comp) + 1):
        # use this i as key, and add to dict
        feature_dict[id] = {}
        id_value = feature_dict[id]  # id_value is a dict
        feature_id_in_next_frame = None  # 1-31, init as None, if exit in next frame change to the id in next frame
        # add appear frame for a feature
        id_value['appear_frame'] = frame_num
        # init disappear frame for a feature
        id_value['disappear_frame'] = None

        img_comp_cur = cur_filtered_comp[i]
        '''results distance'''
        # find the predicted_mask flux of an element
        object_element_flux_1 = cur_flux_dict[i]
        points_contour = cur_contour_dict[i]
        # find the center of component
        center = np.round(img_comp_cur.mean(0))
        '''find roi code'''
        roi = RoI(cur_original_image, center, points_contour, cadence, pixel_resolution)
        roi_x_range = roi[0]
        roi_y_range = roi[1]
        key_flag = False  # decide whether the process of this element finished or not

        '''find element region'''
        e_region = element_region(cur_original_image, center, points_contour)
        e_region_x_range = e_region[0]
        e_region_y_range = e_region[1]

        # find the elements in this roi in the next image
        object_element_roi_next_image = []  # record the element in the next image in roi
        for j in range(1, len(next_filtered_comp) + 1):
            comp_element_2 = next_filtered_comp[j]
            element_contour_points = next_contour_dict[j]
            e_region_flag = False
            for p in element_contour_points:
                if e_region_x_range[0] <= p[0] <= e_region_x_range[1] and e_region_y_range[0] <= p[1] <= \
                        e_region_y_range[1]:
                    e_region_flag = True
                    # non_overlap_flag = False
                    break
            if e_region_flag == True:
                # add a new overlap constrains to solve too much flux change of an element
                if check_with_threshold(img_comp_cur, comp_element_2, overlap_threshold):
                    if check_with_threshold_2(img_comp_cur, comp_element_2, dif_threshold):
                        if check_same_pos_neg(img_comp_cur, comp_element_2, cur_mask_3, next_mask_3):
                            find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
                            if label_option == 0:
                                pass
                            elif label_option == 1:
                                put_num_on_element(cur_original_image, img_comp_cur, i)
                            elif label_option == 2:
                                pass
                            key_flag = True
                            feature_id_in_next_frame = j
                            break
            roi_flag = False
            for p in element_contour_points:
                if roi_x_range[0] <= p[0] <= roi_x_range[1] and roi_y_range[0] <= p[1] <= roi_y_range[1]:
                    roi_flag = True
                    break
            if roi_flag == True:
                flux = next_flux_dict[j]
                theta = abs((object_element_flux_1 - flux) / object_element_flux_1)
                if theta <= 0.5:
                    find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
                    if label_option == 0:
                        pass
                    elif label_option == 1:
                        put_num_on_element(cur_original_image, img_comp_cur, i)
                    elif label_option == 2:
                        pass
                    key_flag = True
                    feature_id_in_next_frame = j
                    break
                object_element_roi_next_image.append(j)  # record the element in the next image in roi

        if key_flag == True:
            if feature_id_in_next_frame not in remain_stable_feature:
                remain_stable_feature[feature_id_in_next_frame] = [id]
            else:
                remain_stable_feature[feature_id_in_next_frame].append(id)
            id += 1
            continue
        else:
            id_value['disappear_frame'] = frame_num
            event_tracking_forward(i, cur_original_image, cur_filtered_comp, next_filtered_comp, cur_flux_dict, next_flux_dict,
            cur_contour_dict, next_contour_dict, cur_mask_3, cadence, pixel_resolution, roi,
                        object_element_roi_next_image, object_element_flux_1, img_comp_cur, points_contour, label_option)
            id += 1
    return remain_stable_feature, feature_dict, id


def backward(remain_stable_feature, id, feature_dict, frame_num, cur_original_image, cur_filtered_comp, previous_filtered_comp, cur_flux_dict, previous_flux_dict, cur_contour_dict,
             previous_contour_dict, cur_mask_3, previous_mask_3, cadence, pixel_resolution, label_option):
    '''process the feature_dict to obtain remain stable feature in previous frame'''
    stable_feature_from_previous = remain_stable_feature.keys()
    overlap_threshold = 0.01
    dif_threshold = 1
    for i in range(1, len(cur_filtered_comp) + 1):
        img_comp_cur = cur_filtered_comp[i]
        object_element_flux_1 = cur_flux_dict[i]
        # check if the feature comes from previous frame
        if i in stable_feature_from_previous:
            previous_id = remain_stable_feature[i]  # previous_id is a list: e.g. [0, 1]
            # loop over the list to find all the global feature id
            for g_id in previous_id:
                global_feature_id = g_id
                id_value = feature_dict[global_feature_id]
                feature_id_in_next_frame = None  # init as None, if exit in next frame change to the id in next frame
                # init disappear frame for a feature, as frame_num, last frame
                id_value['disappear_frame'] = frame_num
        else:
            global_feature_id = id
            feature_dict[global_feature_id] = {}
            id_value = feature_dict[global_feature_id]
            feature_id_in_next_frame = None  # 1 init as None, if exit in next frame change to the id in next frame
            # add appear frame for a feature
            id_value['appear_frame'] = frame_num  # new id need to add frame_num
            # init disappear frame for a feature, as frame_num, last frame
            id_value['disappear_frame'] = frame_num
            id += 1
        # find the center of component
        center = np.round(img_comp_cur.mean(0))
        points_contour = cur_contour_dict[i]
        #'''find roi code'''
        roi = RoI(cur_original_image, center, points_contour, cadence, pixel_resolution)
        roi_x_range = roi[0]
        roi_y_range = roi[1]
        key_flag = False  # decide whether the process of this element finished or not
        #'''find element region'''
        e_region = element_region(cur_original_image, center, points_contour)
        e_region_x_range = e_region[0]
        e_region_y_range = e_region[1]
        # find the elements in this roi in the next image
        object_element_roi_previous_image = []  # record the element in the next image in roi
        for j in range(1, len(previous_filtered_comp) + 1):
            comp_element_2 = previous_filtered_comp[j]
            element_contour_points = previous_contour_dict[j]
            e_region_flag = False
            for p in element_contour_points:
                if e_region_x_range[0] <= p[0] <= e_region_x_range[1] and e_region_y_range[0] <= p[1] <= \
                        e_region_y_range[1]:
                    e_region_flag = True
                    break
            if e_region_flag == True:
                # add a new overlap constrains to solve too much flux change of an element
                if check_with_threshold(img_comp_cur, comp_element_2, overlap_threshold):
                    if check_with_threshold_2(img_comp_cur, comp_element_2, dif_threshold):
                        if check_same_pos_neg(img_comp_cur, comp_element_2, cur_mask_3, previous_mask_3):
                            find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
                            if label_option == 0:
                                pass
                            elif label_option == 1:
                                put_num_on_element(cur_original_image, img_comp_cur, i)
                            elif label_option == 2:
                                pass
                            key_flag = True
                            break
            roi_flag = False
            for p in element_contour_points:
                if roi_x_range[0] <= p[0] <= roi_x_range[1] and roi_y_range[0] <= p[1] <= roi_y_range[1]:
                    roi_flag = True
                    break
            if roi_flag == True:
                flux = previous_flux_dict[j]
                theta = abs((object_element_flux_1 - flux) / object_element_flux_1)
                if theta <= 0.5:
                    find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
                    if label_option == 0:
                        pass
                    elif label_option == 1:
                        put_num_on_element(cur_original_image, img_comp_cur, i)
                    elif label_option == 2:
                        pass
                    key_flag = True
                    break
                object_element_roi_previous_image.append(j)  # record the element in the next image in roi

        if key_flag == True:
            continue
        else:
            event_tracking_backward(i, cur_original_image, cur_filtered_comp, previous_filtered_comp, cur_flux_dict,
                                    previous_flux_dict,
                                    cur_contour_dict, previous_contour_dict, cur_mask_3, cadence, pixel_resolution,
                                    roi, object_element_roi_previous_image, object_element_flux_1, img_comp_cur,
                                    points_contour, label_option, flag_half=False)

    return remain_stable_feature, feature_dict, id


def two_way(remain_stable_feature, id, feature_dict,frame_num, cur_original_image, cur_filtered_comp, next_filtered_comp, previous_filtered_comp, cur_flux_dict, next_flux_dict, previous_flux_dict,
            cur_contour_dict, next_contour_dict, previous_contour_dict, cur_mask_3, next_mask_3, previous_mask_3, cadence, pixel_resolution, label_option):
    '''process the feature_dict to obtain remain stable feature in previous frame'''
    stable_feature_from_previous = remain_stable_feature.keys()
    remain_stable_feature_new = {}
    overlap_threshold = 0.01
    dif_threshold = 1

    for i in range(1, len(cur_filtered_comp) + 1):
        feature_id_in_next_frame = None
        global_feature_id_list = []
        # check if the feature comes from previous frame
        if i in stable_feature_from_previous:
            previous_id = remain_stable_feature[i]
            for g_id in previous_id:
                global_feature_id = g_id
                global_feature_id_list.append(global_feature_id) # add global id to list for later usage
                id_value = feature_dict[global_feature_id]
        else:
            global_feature_id = id
            global_feature_id_list.append(global_feature_id)
            feature_dict[global_feature_id] = {}
            id_value = feature_dict[global_feature_id]
            # add appear frame for a feature
            id_value['appear_frame'] = frame_num  # new id need to add frame_num
            id += 1
        start_time = time.time()
        img_comp_cur = cur_filtered_comp[i]
        object_element_flux_1 = cur_flux_dict[i]

        # find the center of component
        center = np.round(img_comp_cur.mean(0))
        points_contour = cur_contour_dict[i]
        #'''find roi code'''
        roi = RoI(cur_original_image, center, points_contour, cadence, pixel_resolution)
        roi_x_range = roi[0]
        roi_y_range = roi[1]
        key_flag_forward = False  # decide whether the process of this element finished or not
        key_flag_backward = False
        #'''find element region'''
        e_region = element_region(cur_original_image, center, points_contour)
        e_region_x_range = e_region[0]
        e_region_y_range = e_region[1]
        #'''process forward'''
        # find the elements in this roi in the next image
        object_element_roi_next_image = []  # record the element in the next image in roi
        for j_next in range(1, len(next_filtered_comp) + 1):
            comp_element_next = next_filtered_comp[j_next]
            element_contour_points = next_contour_dict[j_next]
            e_region_flag_forward = False
            for p in element_contour_points:
                if e_region_x_range[0] <= p[0] <= e_region_x_range[1] and e_region_y_range[0] <= p[1] <= \
                        e_region_y_range[1]:
                    e_region_flag_forward = True
                    break
            if e_region_flag_forward == True:
                # 3-2-2019: add a new overlap constrains to solve too much flux change of an element
                if check_with_threshold(img_comp_cur, comp_element_next, overlap_threshold):
                    if check_with_threshold_2(img_comp_cur, comp_element_next, dif_threshold):
                        if check_same_pos_neg(img_comp_cur, comp_element_next, cur_mask_3, next_mask_3):
                            key_flag_forward = True
                            feature_id_in_next_frame = j_next
                            break
            roi_flag_forward = False
            for p in element_contour_points:
                if roi_x_range[0] <= p[0] <= roi_x_range[1] and roi_y_range[0] <= p[1] <= roi_y_range[1]:
                    roi_flag_forward = True
                    break
            if roi_flag_forward == True:
                flux = next_flux_dict[j_next]
                theta = abs((object_element_flux_1 - flux) / object_element_flux_1)
                # print(theta)
                if theta <= 0.5:  # suggested 0.05
                    key_flag_forward = True  # True means find an element in the roi reuslts remain stable
                    feature_id_in_next_frame = j_next
                    break
                object_element_roi_next_image.append(j_next)  # record the element in the next image in roi
        # process backward: find the elements in this roi in the previous image
        object_element_roi_previous_image = []  # record the element in the next image in roi
        for j_pre in range(1, len(previous_filtered_comp) + 1):
            comp_element_previous = previous_filtered_comp[j_pre]
            element_contour_points = previous_contour_dict[j_pre]
            e_region_flag_backward = False
            for p in element_contour_points:
                if e_region_x_range[0] <= p[0] <= e_region_x_range[1] and e_region_y_range[0] <= p[1] <= \
                        e_region_y_range[1]:
                    e_region_flag_backward = True
                    break
            if e_region_flag_backward == True:
                # add a new overlap constrains to solve too much flux change of an element
                if check_with_threshold(img_comp_cur, comp_element_previous, overlap_threshold):
                    if check_with_threshold_2(img_comp_cur, comp_element_previous, dif_threshold):
                        if check_same_pos_neg(img_comp_cur, comp_element_previous, cur_mask_3, previous_mask_3):
                            key_flag_backward = True
                            break
            roi_flag_backward = False
            for p in element_contour_points:
                if roi_x_range[0] <= p[0] <= roi_x_range[1] and roi_y_range[0] <= p[1] <= roi_y_range[1]:
                    roi_flag_backward = True
                    break
            if roi_flag_backward == True:
                flux = previous_flux_dict[j_pre]
                theta = abs((object_element_flux_1 - flux) / object_element_flux_1)
                if theta <= 0.5:
                    key_flag_backward = True
                    break
                object_element_roi_previous_image.append(j_pre)  # record the element in the next image in roi
        if key_flag_forward and key_flag_backward:
            find_and_draw_contour_line_new(cur_original_image, points_contour, cur_mask_3)
            if label_option == 0:
                pass
            elif label_option == 1:
                put_num_on_element(cur_original_image, img_comp_cur, i)
            elif label_option == 2:
                pass

            for global_feature_id in global_feature_id_list:
                id_value = feature_dict[global_feature_id]
                if feature_id_in_next_frame not in remain_stable_feature_new:
                    remain_stable_feature_new[feature_id_in_next_frame] = [global_feature_id]
                else:
                    remain_stable_feature_new[feature_id_in_next_frame].append(global_feature_id)
            continue
        elif not key_flag_forward and key_flag_backward:
            for global_feature_id in global_feature_id_list:
                id_value = feature_dict[global_feature_id]
                id_value['disappear_frame'] = frame_num
            event_tracking_forward(i, cur_original_image, cur_filtered_comp, next_filtered_comp, cur_flux_dict,
                                   next_flux_dict,
                                   cur_contour_dict, next_contour_dict, cur_mask_3, cadence, pixel_resolution, roi,
                                   object_element_roi_next_image, object_element_flux_1, img_comp_cur, points_contour,
                                   label_option)
        elif not key_flag_backward and key_flag_forward:
            for global_feature_id in global_feature_id_list:
                id_value = feature_dict[global_feature_id]
                # id_value['feature_id_in_next_frame'] = feature_id_in_next_frame
                if feature_id_in_next_frame not in remain_stable_feature_new:
                    remain_stable_feature_new[feature_id_in_next_frame] = [global_feature_id]
                else:
                    remain_stable_feature_new[feature_id_in_next_frame].append(global_feature_id)

            event_tracking_backward(i, cur_original_image, cur_filtered_comp, previous_filtered_comp, cur_flux_dict,
                                    previous_flux_dict,
                                    cur_contour_dict, previous_contour_dict, cur_mask_3, cadence, pixel_resolution,
                                    roi, object_element_roi_previous_image, object_element_flux_1, img_comp_cur,
                                    points_contour,label_option, flag_half=False)

        else:
            for global_feature_id in global_feature_id_list:
                id_value = feature_dict[global_feature_id]
                id_value['disappear_frame'] = frame_num
            event_tracking_forward(i, cur_original_image, cur_filtered_comp, next_filtered_comp, cur_flux_dict,
                                   next_flux_dict,
                                   cur_contour_dict, next_contour_dict, cur_mask_3, cadence, pixel_resolution, roi,
                                   object_element_roi_next_image, object_element_flux_1, img_comp_cur, points_contour,
                                   label_option)
            event_tracking_backward(i, cur_original_image, cur_filtered_comp, previous_filtered_comp, cur_flux_dict,
                                    previous_flux_dict,
                                    cur_contour_dict, previous_contour_dict, cur_mask_3, cadence, pixel_resolution,
                                    roi, object_element_roi_previous_image, object_element_flux_1, img_comp_cur,
                                    points_contour, label_option, flag_half=True)

    return remain_stable_feature_new, feature_dict, id


def magnetic_tracking(input_path, output_path, save_lifetime=None):
    print('============magnetic tracking start============')
    # '''read the data'''
    fits_files_dict, image_dict, mask_2_class_dict, mask_3_class_dict, num_frames = read_data_files(input_path)

    cadence = 56   # 72 for 2018; 56 for 2017
    pixel_resolution = 0.083
    label_option = 0  # 0: label;  1: no label
    start = 1  # start tracking frame

    filtered_comp_dict, flux_dict, contour_dict = pre_calulate_flux_and_contour(fits_files_dict, mask_2_class_dict, num_frames)

    total_ts = time.time()
    feature_dict = {}
    remain_stable_feature = {}
    id = 1
    # '''process each frame'''
    for i in range(start, num_frames + 1):
        print('-----------------process frame {}------------------'.format(i))
        cur_original_image = image_dict[i]
        cur_mask_3_class = mask_3_class_dict[i]
        cur_filtered_comp = filtered_comp_dict[i]
        cur_flux_dict = flux_dict[i]
        cur_contour_dict = contour_dict[i]
        # i is frame number
        if i == start:
            next_mask_3_class = mask_3_class_dict[i+1]
            next_filtered_comp = filtered_comp_dict[i+1]
            next_flux_dict = flux_dict[i+1]
            next_contour_dict = contour_dict[i+1]

            remain_stable_feature, feature_dict, id = forward(remain_stable_feature,id, feature_dict, i, cur_original_image, cur_filtered_comp, next_filtered_comp, cur_flux_dict, next_flux_dict, cur_contour_dict,
                    next_contour_dict, cur_mask_3_class, next_mask_3_class, cadence, pixel_resolution, label_option)

            cv2.imwrite('{}tracking_result_frame_{}.png'.format(output_path, i), cur_original_image)
        elif i == num_frames:
            previous_mask_3_class = mask_3_class_dict[i-1]
            previous_filtered_comp = filtered_comp_dict[i-1]
            previous_flux_dict = flux_dict[i-1]
            previous_contour_dict = contour_dict[i-1]

            remain_stable_feature, feature_dict, id = backward(remain_stable_feature, id, feature_dict, i, cur_original_image, cur_filtered_comp, previous_filtered_comp, cur_flux_dict, previous_flux_dict, cur_contour_dict,
                     previous_contour_dict, cur_mask_3_class, previous_mask_3_class, cadence, pixel_resolution, label_option)

            cv2.imwrite('{}tracking_result_frame_{}.png'.format(output_path, i), cur_original_image)
        else:
            next_mask_3_class = mask_3_class_dict[i+1]
            next_filtered_comp = filtered_comp_dict[i+1]
            next_flux_dict = flux_dict[i+1]
            next_contour_dict = contour_dict[i+1]

            previous_mask_3_class = mask_3_class_dict[i-1]
            previous_filtered_comp = filtered_comp_dict[i-1]
            previous_flux_dict = flux_dict[i-1]
            previous_contour_dict = contour_dict[i-1]

            remain_stable_feature, feature_dict, id = two_way(remain_stable_feature, id, feature_dict, i, cur_original_image, cur_filtered_comp, next_filtered_comp, previous_filtered_comp, cur_flux_dict,
                    next_flux_dict, previous_flux_dict,
                    cur_contour_dict, next_contour_dict, previous_contour_dict, cur_mask_3_class, next_mask_3_class,
                    previous_mask_3_class, cadence, pixel_resolution, label_option)

            cv2.imwrite('{}tracking_result_frame_{}.png'.format(output_path, i), cur_original_image)
    #
    total_te = time.time()
    print('-----------------Done------------------')
    # print('-------------------time cost--------------------{}'.format(total_te - total_ts))

    # find the lifetime information of all the features
    if save_lifetime:
        feature_lifetime_list = []
        for key in feature_dict:
            value = feature_dict[key]
            appear_frame = value['appear_frame']
            disappear_frame = value['disappear_frame']
            exist_frame_num = disappear_frame - appear_frame + 1

            feature_lifetime_list.append(exist_frame_num)
        # write to file
        with open('{}feature_lifetime.csv'.format(save_lifetime), 'w') as myFile:
            writer = csv.writer(myFile)
            writer.writerow(feature_lifetime_list)
