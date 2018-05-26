import random
import numpy as np
from sklearn.utils import shuffle
from data_utils import *
import matplotlib.pyplot as plt
from collections import defaultdict
def apply_transform(x,transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def augment_image(x,rotation_range=0,height_shift_range=0,
                  width_shift_range=0,img_row_axis=1,img_col_axis=2,
                  img_channel_axis=0,horizontal_flip=False,vertical_flip=False):

    if rotation_range != 0:
        theta = np.deg2rad(np.random.uniform(rotation_range, rotation_range))
    else:
        theta = 0

    if height_shift_range != 0:
        tx = np.random.uniform(-height_shift_range, height_shift_range)
        if height_shift_range < 1:
            tx *= x.shape[img_row_axis]
    else:
        tx = 0

    if width_shift_range != 0:
        ty = np.random.uniform(-width_shift_range, width_shift_range)
        if width_shift_range < 1:
            ty *= x.shape[img_col_axis]
    else:
        ty = 0

    transform_matrix = None
    if theta != 0:
        transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])
    if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_axis)

    if vertical_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_axis)

    if transform_matrix is not None:
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis)

    return x
def augment_batch(images,rotation_range=0,height_shift_range=0,
                  width_shift_range=0,img_row_axis=1,img_col_axis=2,
                  img_channel_axis=0,horizontal_flip=False,vertical_flip=False):
    x = np.array(images)
    indx = images.shape[0]
    for i in range(0,indx):
        x[i] = augment_image(images[i],
         rotation_range=rotation_range,
         img_row_axis=img_row_axis,
         img_col_axis=img_col_axis,
         img_channel_axis=img_channel_axis,
         horizontal_flip = horizontal_flip,
         vertical_flip = vertical_flip,
         height_shift_range = height_shift_range,
         width_shift_range = width_shift_range)
    return x