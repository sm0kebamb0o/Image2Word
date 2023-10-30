from os import path, remove
import cv2 as cv
import numpy as np
from enum import Enum
import sys

import config
import utils
from deslant_img import deslant_img
from processing import WordProcessor, PositionMode

'''
class DataNormalizer:
    """
    Extract labels from words.txt, erase invalid images and
    create new dataset with normalized pictures.
    """

    def __init__(self,
                 raw_labels_file: str,
                 labels_file: str,
                 image_height: int,
                 image_width: int):
        """
        Keyword arguments:
        raw_labels_file: the IAM file with the information about images
        labels_file: the file, where te resulting labels would be stored
        image_height: the required height of all the images
        image_width: the reauired width of all the imagess
        """
        self.raw_labels_file = raw_labels_file
        self.labels_file = labels_file
        self.preproccesor = WordProcessor(image_height=image_height,
                                          image_width=image_width,
                                          position_mode=PositionMode.Left)
        self.images_number = 0

    def __call__(self) -> dict:
        """Returns a map with terminals and their id."""

        dictionary = dict()
        cur_label_id = 1

        with open(self.raw_labels_file, 'r') as raw_labels, open(self.labels_file, 'w') as labels:
            for line in raw_labels:
                if line[0] == '#':
                    continue

                line = line.split()

                image_path_parts = line[0].split('-')
                image_path = path.join(
                    self.dir_path,
                    'words',
                    image_path_parts[0],
                    '-'.join(image_path_parts[:2]), line[0] + '.png')

                if not path.isfile(image_path):
                    continue

                label = ' '.join(line[8:])

                if path.getsize(image_path) == 0 or line[1] != "ok":
                    remove(image_path)
                    continue

                label = self.__make_valid_label(label, config.MAX_LABEL_LENGTH)

                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

                images = self.preproccesor(image)
                images_paths = [image_path[:-4] +
                                f"_{i}.png" for i in range(1, len(images)+1)]
                self.images_number = len(images)

                for image, image_path in zip(images, images_paths):
                    cv.imwrite(image_path, image)
                    print(image_path, label, file=labels)

                for symbol in list(label):
                    if symbol not in dictionary:
                        dictionary[symbol] = cur_label_id
                        cur_label_id += 1
            return dictionary

    def __make_valid_label(self, label: str, max_len: int) -> str:
        """Cuts the label so that would fit in required size in CTCLoss terms."""
        cur_len = 1
        for i in range(1, len(label)):
            if label[i] == label[i - 1]:
                # Here we are adding 2, because between same symbols
                # there should be a special blank
                cur_len += 2
            else:
                cur_len += 1
            if cur_len > max_len:
                return label[:i]
        return label

def divide_dataset(labels_file: str,
                   train_file: str,
                   val_file: str,
                   test_file: str,
                   same_images: int,
                   validation_percent:float,
                   testing_parcent:float):
    """
    Divide dataset into three parts: training, validation, testing

    Keyword arguments:
    dir_path: the directory, where labels and images have been saved
    labels_file: the file, where labels are stored
    train_file: the file, where labels for training would be saved
    val_file: the file, where labels for validation would be stored
    test_file: the file, where labels for testing would be stored
    same_images: number of preprocessed images that correspond to one initial
    """
    assert 0. <= validation_percent <= 1.
    assert 0. <= testing_parcent <= 1.
    assert validation_percent + testing_parcent <= 1.
    
    with open(labels_file, 'r') as fin,      \
            open(train_file, 'w') as ftrain, \
            open(val_file, 'w') as fval,     \
            open(test_file, 'w') as ftest:
        lines = fin.readlines()
        ids = [i for i in range(0, len(lines))]

        val_num = int(len(ids) * validation_percent)
        test_num = int(len(ids) * testing_parcent)
        train_num = len(ids) - val_num - test_num

        train = np.random.choice(ids, size=train_num, replace=False)
        ids = np.setdiff1d(ids, train)

        val = np.random.choice(ids, size=val_num, replace=False)
        test = np.setdiff1d(ids, val)

        def write_pathes_to_file(ids, ffile):
            for id in ids:
                for i in range(1, same_images+1):
                    print(lines[id].replace('.png', f"_{i}.png"), file=ffile, end='')
        
        write_pathes_to_file(train, ftrain)
        write_pathes_to_file(val, fval)
        write_pathes_to_file(test, ftest)
'''

class DataNormalizer:
    """
    Extract labels from words.txt, erase invalid images and
    create new dataset with normalized pictures.
    """

    def __init__(self,
                 data_dir:str,
                 raw_labels_file: str,
                 labels_file: str):
        """
        Keyword arguments:
        raw_labels_file: the IAM file with the information about images
        labels_file: the file, where te resulting labels would be stored
        image_height: the required height of all the images
        image_width: the reauired width of all the imagess
        """
        self.data_dir = data_dir
        self.raw_labels_file = raw_labels_file
        self.labels_file = labels_file

    def __call__(self) -> dict:
        """Returns a map with terminals and their id."""

        dictionary = dict()
        cur_label_id = 1

        with open(self.raw_labels_file, 'r') as raw_labels, open(self.labels_file, 'w') as labels:
            for line in raw_labels:
                if line[0] == '#':
                    continue

                line = line.split()

                image_path_parts = line[0].split('-')
                image_path = path.join(
                    self.data_dir,
                    'words',
                    image_path_parts[0],
                    '-'.join(image_path_parts[:2]), line[0] + '.png')

                if not path.isfile(image_path):
                    continue

                label = ' '.join(line[8:])

                if path.getsize(image_path) == 0 or line[1] != "ok":
                    remove(image_path)
                    continue

                label = self.__make_valid_label(label, config.MAX_LABEL_LENGTH)

                print(image_path, label, file=labels)

                for symbol in list(label):
                    if symbol not in dictionary:
                        dictionary[symbol] = cur_label_id
                        cur_label_id += 1
            return dictionary

    def __make_valid_label(self, label: str, max_len: int) -> str:
        """Cuts the label so that would fit in required size in CTCLoss terms."""
        cur_len = 1
        for i in range(1, len(label)):
            if label[i] == label[i - 1]:
                # Here we are adding 2, because between same symbols
                # there should be a special blank
                cur_len += 2
            else:
                cur_len += 1
            if cur_len > max_len:
                return label[:i]
        return label

def divide_dataset(labels_file: str,
                   train_file: str,
                   val_file: str,
                   test_file: str,
                   validation_percent:float,
                   testing_parcent:float):
    """
    Divide dataset into three parts: training, validation, testing

    Keyword arguments:
    labels_file: the file, where labels are stored
    train_file: the file, where labels for training would be saved
    val_file: the file, where labels for validation would be stored
    test_file: the file, where labels for testing would be stored
    same_images: number of preprocessed images that correspond to one initial
    """
    assert 0. <= validation_percent <= 1.
    assert 0. <= testing_parcent <= 1.
    assert validation_percent + testing_parcent <= 1.
    
    with open(labels_file, 'r') as fin,      \
            open(train_file, 'w') as ftrain, \
            open(val_file, 'w') as fval,     \
            open(test_file, 'w') as ftest:
        lines = fin.readlines()
        ids = [i for i in range(0, len(lines))]

        val_num = int(len(ids) * validation_percent)
        test_num = int(len(ids) * testing_parcent)
        train_num = len(ids) - val_num - test_num

        train = np.random.choice(ids, size=train_num, replace=False)
        ids = np.setdiff1d(ids, train)

        val = np.random.choice(ids, size=val_num, replace=False)
        test = np.setdiff1d(ids, val)

        for id in train:
            print(lines[id], file=ftrain, end='')
        for id in val:
            print(lines[id], file=fval, end='')
        for id in test:
            print(lines[id], file=ftest, end='')


if __name__ == '__main__':
    '''
    normalizer = DataNormalizer(data_dir=config.DATA_PATH,
                                raw_labels_file=path.join(config.DATA_PATH, config.RAW_LABELS_FILE),
                                labels_file=path.join(config.DATA_PATH, config.LABELS_FILE))
    chars_used = normalizer()
    print('Characters appeared in dataset:')
    print(chars_used)
    '''

    divide_dataset(labels_file=path.join(config.DATA_PATH, config.LABELS_FILE),
                   train_file=path.join(config.DATA_PATH, config.TRAIN_FILE),
                   val_file=path.join(config.DATA_PATH, config.VAL_FILE),
                   test_file=path.join(config.DATA_PATH, config.TEST_FILE),
                   validation_percent=config.VALIDATION_PERCENT,
                   testing_parcent=config.TESTING_PERCENT)
