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

import warnings
import os
import sys
warnings.filterwarnings('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    # print('turning logging of is not available')
    pass
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import cv2
import matplotlib
import skimage.io as io
import skimage.transform as trans
from astropy.io import fits
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
sys.stderr = stderr
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcolor


def conv2_block(input_tensor, n_filters):
    x = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return x


def solarUnet(pretrained_weights=None, n_filters=32, input_size=(720, 720, 1)):
    inputs = Input(input_size)
    conv1 = conv2_block(inputs, n_filters * 1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = conv2_block(drop1, n_filters * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = conv2_block(drop2, n_filters * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)

    conv4 = conv2_block(drop3, n_filters * 8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)

    conv5 = conv2_block(drop4, n_filters * 16)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(n_filters * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    drop6 = Dropout(0.5)(merge6)
    conv6 = conv2_block(drop6, n_filters * 8)

    up7 = Conv2D(n_filters * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    drop7 = Dropout(0.5)(merge7)
    conv7 = conv2_block(drop7, n_filters * 4)

    up8 = Conv2D(n_filters * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    drop8 = Dropout(0.5)(merge8)
    conv8 = conv2_block(drop8, n_filters * 2)

    up9 = Conv2D(n_filters * 1, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    drop9 = Dropout(0.5)(merge9)
    conv9 = conv2_block(drop9, n_filters * 1)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def adjust_data(img, mask):
    if np.max(img) > 1:
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def train_generator(batch_size, train_path, image_folder, mask_folder):

    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(720,720),
        batch_size=batch_size,
        seed=1)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(720,720),
        batch_size=batch_size,
        seed=1)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img,mask = adjust_data(img, mask)
        yield img, mask


def validation_generator(batch_size, train_path,image_folder, mask_folder):

    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(720,720),
        batch_size=batch_size,
        seed=1)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=(720,720),
        batch_size=batch_size,
        seed=1)
    validation_generator = zip(image_generator, mask_generator)
    for (img, mask) in validation_generator:
        img,mask = adjust_data(img, mask)
        yield img, mask


def test_generator(test_path, target_size=(720,720)):
    for name in os.listdir(test_path):
        img = io.imread(os.path.join(test_path, name), as_gray=True)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img


def save_result(save_path, result):
    for i, item in enumerate(result):
        img = np.round(item) * 255
        cv2.imwrite(os.path.join(save_path, "predicted_mask_{0:03}.png".format(i)), img)


def pre_processing(input_3_class_mask_path, output_path):
    """
    convert SWAMIS 3-class maks to 2-class
    user may provide their own path
    input_3_class_mask_path = 'SWAMIS_mask_path'
    output_path = 'output_path'
    """
    '''read the corresponding fits data'''
    for name in os.listdir(input_3_class_mask_path):
        print(name)
        fits_file = fits.open(input_3_class_mask_path + name)
        fits_data = fits_file[0].data
        fits_data = np.flipud(fits_data)
        height, width = fits_data.shape

        output_image = np.empty([height, width])
        output_image.fill(np.nan)
        for i in range(height):
            for j in range(width):
                if fits_data[i][j] == 0:
                    output_image[i][j] = 255
                else:
                    output_image[i][j] = 0
        cv2.imwrite('{}{}.png'.format(output_path, name[:-4]), output_image)


def post_processing():
    """read corresponding predicted_mask field data"""
    predicted_mask_2_path = 'results/predicted_mask/'
    output_mask_2_path = 'results/processed_data_for_tracking/mask_2_class/'
    output_mask_3_path = 'results/processed_data_for_tracking/mask_3_class/'
    fitz_path = 'results/processed_data_for_tracking/fits/'

    predicted_mask_2_files_list = os.listdir(predicted_mask_2_path)

    file_index = 0
    for name in os.listdir(fitz_path):

        fits_file = fits.open(fitz_path+name)
        fits_data = fits_file[0].data
        fits_data = np.flipud(fits_data)
        height, width = fits_data.shape
        fits_file.close()

        image_name = '{}{}'.format(predicted_mask_2_path,predicted_mask_2_files_list[file_index])
        file_index += 1
        img = cv2.imread(image_name, -1)

        output_image = np.empty([height, width])
        output_image.fill(np.nan)

        for i in range(height):
            for j in range(width):
                if img[i][j] == 255:
                    output_image[i][j] = 127
                elif img[i][j] == 0:
                    if fits_data[i][j] < -150:
                        output_image[i][j] = 0
                    elif fits_data[i][j] > 150:
                        output_image[i][j] = 255
                    else:
                        output_image[i][j] = 127
                else:
                    if fits_data[i][j] < -150:
                        output_image[i][j] = 0
                    elif fits_data[i][j] > 150:
                        output_image[i][j] = 255
                    else:
                        output_image[i][j] = 127
        mask_3 = output_image

        # remove connection
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                p = mask_3[i][j]
                p1 = mask_3[i - 1][j - 1]
                p2 = mask_3[i - 1][j]
                p3 = mask_3[i - 1][j + 1]
                p4 = mask_3[i][j - 1]
                p5 = mask_3[i][j + 1]
                p6 = mask_3[i + 1][j - 1]
                p7 = mask_3[i + 1][j]
                p8 = mask_3[i + 1][j + 1]
                if p != 127:
                    if p == 255:
                        if 0 in [p1, p2, p3, p4, p5, p6, p7, p8]:
                            mask_3[i][j] = 127
                    elif p == 0:
                        if 255 in [p1, p2, p3, p4, p5, p6, p7, p8]:
                            mask_3[i][j] = 127

        cv2.imwrite(output_mask_3_path+'mask_3_class_{0:03}.png'.format(file_index), mask_3)

        for i in range(height):
            for j in range(width):
                if mask_3[i][j] == 127:
                    mask_3[i][j] = 255  # 0 black,   for object detection forgroud should be white , backgroud should black
                else:
                    mask_3[i][j] = 0  # 255 white
        cv2.imwrite(output_mask_2_path + 'mask_2_class_{0:03}.png'.format(file_index), mask_3)
    # print('Postprocess done')


def plot_mask():
    predicted_mask_path = 'results/processed_data_for_tracking/mask_3_class/'
    testing_image_path = 'data/magnetic/test/'
    output_path = 'results/demo_plots/'

    img = mpimg.imread(testing_image_path + 'frame_1.png')
    mask = mpimg.imread(predicted_mask_path + 'mask_3_class_001.png')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax[0].imshow(img, cmap='gray', extent=(413, 473, -204, -144))
    ax[0].set_xlabel("E-W (arcsec)", fontsize=10)
    ax[0].set_ylabel("S-N (arcsec)", fontsize=10)
    ax[0].set_title('Testing Magnetogram', fontsize=12)

    ax[1].imshow(mask, cmap='gray', extent=(413, 473, -204, -144))
    ax[1].set_xlabel("E-W (arcsec)", fontsize=10)
    ax[1].set_ylabel("S-N (arcsec)", fontsize=10)
    ax[1].set_title('SolarUnet Mask', fontsize=12)

    plt.savefig(output_path+'demo_3_class_masks.png', bbox_inches='tight')
    plt.show()


def plot_tracking_results():
    extent_value = (413, 473, -204, -144)
    input_path = 'results/tracking_results/'
    output_path = 'results/demo_plots/'
    img_list = []
    for name in os.listdir(input_path):
        img = mpimg.imread(input_path + name)
        img_list.append(img)

    img1 = img_list[0]
    img2 = img_list[1]
    img3 = img_list[2]
    my_map = 'gray'
    fontsize = 10
    norm = mpcolor.Normalize(vmin=-500, vmax=500)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    im = ax[0].imshow(img1, cmap=my_map, extent=extent_value, norm=norm)
    ax[0].set_xlabel("E-W (arcsec)", fontsize=fontsize)
    ax[0].set_ylabel("S-N (arcsec)", fontsize=fontsize)
    ax[0].set_title('frame-1', x=0.9, y=0, fontsize=fontsize)

    ax[1].imshow(img2, cmap=my_map, extent=extent_value, norm=norm)
    ax[1].set_xlabel("E-W (arcsec)", fontsize=fontsize)
    ax[1].set_ylabel("S-N (arcsec)", fontsize=fontsize)
    ax[1].set_title('frame-2', x=0.9, y=0, fontsize=fontsize)

    ax[2].imshow(img3, cmap=my_map, extent=extent_value, norm=norm)
    ax[2].set_xlabel("E-W (arcsec)", fontsize=fontsize)
    ax[2].set_ylabel("S-N (arcsec)", fontsize=fontsize)
    ax[2].set_title('frame-3', x=0.9, y=0, fontsize=fontsize)

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.15)
    cax = plt.axes((0.92, 0.168, 0.014, 0.665))
    cb = fig.colorbar(im, cax=cax, norm=norm)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label('Gauss', fontsize=fontsize)

    plt.savefig(output_path+'demo_tracking_result.png', bbox_inches='tight')
    plt.show()


def model_training(input_path):
    """for model training
    input path: 'data/magnetic/'
    output path: 'results/predicted_mask/'
    """
    train_datagen = train_generator(1, input_path+'train', 'image', 'label')
    model = solarUnet()
    model_checkpoint = ModelCheckpoint('solarunet_magnetic.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(train_datagen, steps_per_epoch=10000, epochs=1, verbose=1, callbacks=[model_checkpoint])


def model_predicting(input_path, output_path, pretrain=False):
    """use trained model to predict predicted_mask"""
    if pretrain:
        model = solarUnet('pretrained_model/solarUnet_magnetic.hdf5')
    else:
        model = solarUnet('solarunet_magnetic.hdf5')
    test_datagen = test_generator(input_path+'test')
    results = model.predict_generator(test_datagen, 3, verbose=1)
    save_result(output_path, results)
    print('Prediction on the given data done')

