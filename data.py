import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

_correction = 0.2

_CAMERA_INDEX_LEFT = 1
_CAMERA_INDEX_RIGHT = 2

def _flip(img):
    return cv2.flip(img, 1)

def _random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    weight = np.random.uniform(0.3, 1.2)
    if weight > 1.0:
        hsv[:, :, 2] = np.minimum(hsv[:, :, 2] * weight, 255)
    else:
        hsv[:, :, 2] = hsv[:, :, 2] * weight

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _random_gamma(img):
    gamma = np.random.uniform(0.3, 1.7)
    # http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def _concat_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.concatenate((img, hsv), 2)

def preprocess(img):
    return _concat_hsv(img)

def _generator(samples, is_train, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # there are 3 camera images (center, left, right) in a row
                cameras = [0]
                if is_train:
                    cameras = range(3)
                for camera_index in cameras:
                    path = batch_sample[camera_index]
                    image = cv2.imread(path)
                    angle = float(batch_sample[3])
                    # needs some angle correction
                    if camera_index == _CAMERA_INDEX_LEFT:
                        angle += _correction
                    elif camera_index == _CAMERA_INDEX_RIGHT:
                        angle -= _correction

                    images.append(image)
                    measurements.append(angle)

            result_images, result_measurements = [], []
            if is_train:
                # to augment the data, add the flipping data
                for image, measurement in zip(images, measurements):
                    result_images.append(image)
                    result_measurements.append(measurement)

                    result_images.append(_random_brightness(image))
                    result_measurements.append(measurement)

                    result_images.append(_random_gamma(image))
                    result_measurements.append(measurement)

                    flipped = _flip(image)
                    result_images.append(flipped)
                    result_measurements.append(measurement * -1.0)

                    result_images.append(_random_brightness(flipped))
                    result_measurements.append(measurement * -1.0)

                    result_images.append(_random_gamma(flipped))
                    result_measurements.append(measurement * -1.0)
            else:
                result_images = images
                result_measurements = measurements

            preprocessed_images = []
            for img in result_images:
                preprocessed_images.append(preprocess(img))

            # trim image to only see section with road
            X_train = np.array(preprocessed_images)
            y_train = np.array(result_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

_DATA_DIR = './data'

_train_data_len_multiply = 3 * 3 * 2 # cameras * random data * flip

def _get_samples():
    data_dirs = []
    for path in os.listdir(_DATA_DIR):
        dir = _DATA_DIR + "/" + path
        if os.path.isdir(dir):
            data_dirs.append(dir)

    samples = []
    for dir in data_dirs:
        with open(dir + "/driving_log.csv") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                for i in range(3):
                    filename = line[i].split('/')[-1]
                    line[i] = dir + '/IMG/' + filename
                samples.append(line)

    return samples

def generators():
    samples = _get_samples()
    from sklearn.model_selection import train_test_split
    tv_samples, test_samples = train_test_split(samples, test_size=0.2)
    train_samples, validation_samples = train_test_split(tv_samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = _generator(train_samples, True, batch_size=64)
    validation_generator = _generator(validation_samples, False, batch_size=64)
    test_generator = _generator(test_samples, False, batch_size=64)
    return train_generator, (len(train_samples) * _train_data_len_multiply), validation_generator, len(validation_samples), test_generator, len(test_samples)
