from os import path, remove
from typing import Any
import cv2 as cv
import numpy as np
from enum import Enum

import config
from deslant_img import deslant_img


class PositionMode(Enum):
    """Stores diferent positioning modes."""
    Left = 0
    Right = 1
    Random = 2

    def __len__(self) -> int:
        return super().__len__()


class DataNormalizer:
    """
    Extract labels from words.txt, erase invalid images and
    create new dataset with normalized pictures.
    """
    def __init__(self, 
                 dir_path: str, 
                 raw_labels_file: str, 
                 labels_file: str, 
                 image_height: int, 
                 image_width: int):
        """
        Keyword arguments:
        dir_path: the directory, where labels and images have been saved
        raw_labels_file: the IAM file with the information about images
        labels_file: the file, where te resulting labels would be stored
        image_height: the required height of all the images
        image_width: the reauired width of all the imagess
        """
        self.dir_path = dir_path
        self.raw_labels_file = raw_labels_file
        self.labels_file = labels_file
        self.preproccesor = DataPreprocessor(image_height=image_height,
                                             image_width=image_width,
                                             position_mode=PositionMode.Left)
        self.images_number = 0

    def __call__(self) -> dict:
        """Returns a map with terminals and their id."""
        raw_labels_file = path.join(self.dir_path, self.raw_labels_file)
        labels_file = path.join(self.dir_path, self.labels_file)

        dictionary = dict()
        cur_label_id = 1

        with open(raw_labels_file, 'r') as raw_labels, open(labels_file, 'w') as labels:
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

                if path.getsize(image_path)==0 or line[1]!="ok" or (len(label)==1 and not label.isalpha()) :
                    remove(image_path)
                    continue

                label = self.__make_valid_label(label, config.MAX_LABEL_LENGTH)

                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                
                images = self.preproccesor(image)
                images_paths = [image_path[:-4] + f"_{i}.png" for i in range(1, len(images)+1)]
                self.images_number = len(images)

                for image, image_path in zip(images, images_paths):
                    cv.imwrite(image_path, image)
                    print(image_path, label, file=labels)

                for symbol in list(label):
                    if symbol not in dictionary:
                        dictionary[symbol] = cur_label_id
                        cur_label_id += 1
            return dictionary

    def __make_valid_label(self, label : str, max_len : int) -> str:
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

class DataPreprocessor:
    def __init__(self, 
                 image_height:int, 
                 image_width:int, 
                 position_mode=PositionMode.Left):
        """
        Keyword arguments:
        image_height: required height of the resulting image
        image_width: required width of the resulting image
        position: place, where to store the input in the resulting image 
                could be 'left', 'right' or 'random' (default 'left')
        """
        self.image_height = image_height
        self.image_width = image_width
        self.position_mode = position_mode
    
    def __call__(self, image:np.ndarray) -> tuple:
        """
        Remove noise from the image, remove slant and bring to the required view.
        Returns tuple of images with different level of noise and intensity.
        """
        assert len(image.shape) == 2

        images = self.__remove_noise(image)

        deslanted_images = [deslant_img(img=image)[0] for image in images]

        return ([self.__resize_image(image) for image in deslanted_images])

    def __resize_image(self, image) -> np.ndarray:
        """
        Brings image to the required size and stores it at the specified 
        position in the resulting image.
        """
        assert len(image.shape) == 2

        (cur_image_height, cur_image_width) = image.shape

        if cur_image_height == self.image_height and cur_image_width == self.image_width:
            return image

        height_rel = cur_image_height / self.image_height
        width_rel = cur_image_width / self.image_width

        if height_rel > width_rel:
            new_image_height = self.image_height
            new_image_width = int(cur_image_width / height_rel)
        else:
            new_image_height = int(cur_image_height / width_rel)
            new_image_width = self.image_width

        image = cv.resize(image, [new_image_width, new_image_height])

        req_image = np.full(
            shape=[self.image_height, self.image_width], fill_value=255, dtype=int)
        
        cur_position_mode = np.random.choice(PositionMode, 1) if self.position_mode == PositionMode.Random \
            else self.position_mode

        if cur_position_mode == PositionMode.Left:
            req_image[:new_image_height, :new_image_width] = image
        elif cur_position_mode == PositionMode.Right:
            req_image[-new_image_height:, -new_image_width:] = image

        return req_image

    def __remove_noise(self, image:np.ndarray) -> tuple:
        """
        Remove all noise from the image. Returns 3 images with different 
        level of noise and intensity.

        Keyword arguments:
        image: image that should be tranformed
        """
        def calc_kernel_size(anchor: int, percent: float) -> int:
            kernel_size = int(percent * anchor)
            kernel_size += ((kernel_size & 1) + 1) & 1
            return kernel_size
        
        def calc_gaussian_kernel_size(anchor: int, percent: float) -> int:
            kernel_size = calc_kernel_size(anchor, percent)
            return kernel_size if kernel_size > 1 else 3

        blur_kernel = (calc_gaussian_kernel_size(
            image.shape[0], 0.075), calc_gaussian_kernel_size(image.shape[1], 0.02))
        image_blur = cv.GaussianBlur(image, ksize=blur_kernel, sigmaX=0)

        _, image_bin = cv.threshold(src=image_blur, 
                                    thresh=220, 
                                    maxval=255,
                                    type=cv.THRESH_OTSU)
        image_bin_help = cv.adaptiveThreshold(src=image_blur, 
                                            maxValue=255, 
                                            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            thresholdType=cv.THRESH_BINARY, 
                                            blockSize=calc_gaussian_kernel_size(
                                                max(image.shape), image.shape[0]/image.shape[1] * 0.25),
                                            C=5)
        image_bin[image_bin_help == 255] = 255

        def rle(row):
            assert len(row.shape) == 1
            y = row[1:] != row[:-1]
            i = np.append(np.where(y), row.shape[0] - 1)
            z = np.diff(np.append(-1, i))
            return (z, row[i])
        
        lines = []
        for row in range(image_bin.shape[0]):
            lengths, elems = rle(image_bin[row, :])
            lines.extend(lengths[elems == 0])
        h_stroke_len = int(np.median(lines))

        columns = []
        for column in range(image_bin.shape[1]):
            lengths, elems = rle(image_bin[:, column])
            columns.extend(lengths[elems == 0])
        v_stroke_len = int(np.median(columns))

        mid_kernel = (calc_kernel_size(h_stroke_len, 0.25),
                    calc_kernel_size(v_stroke_len, 0.25))
        small_kernel = (calc_kernel_size(h_stroke_len, 0.2),
                        calc_kernel_size(v_stroke_len, 0.2))
        
        image_border = cv.morphologyEx(src=image_bin, 
                                    op=cv.MORPH_GRADIENT, 
                                    kernel=cv.getStructuringElement(shape=cv.MORPH_RECT, 
                                                                    ksize=mid_kernel))
        image_border += image_bin

        image_eroded = cv.morphologyEx(
            image_border, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_RECT, mid_kernel))

        image_eroded = cv.morphologyEx(image_eroded, cv.MORPH_ERODE,
                                    kernel=cv.getStructuringElement(cv.MORPH_RECT, small_kernel))
        
        return (image_bin, image_border, image_eroded)
    
def divide_dataset(dir_path:str,
                   labels_file:str,
                   train_file:str,
                   val_file:str,
                   test_file:str,
                   same_images:int):
    """
    Keyword arguments:
    dir_path: the directory, where labels and images have been saved
    labels_file: the file, where labels are stored
    train_file: the file, where labels for training would be saved
    val_file: the file, where labels for validation would be stored
    test_file: the file, where labels for testing would be stored
    same_images: number of preprocessed images that correspond to one initial
    """
    labels_file = path.join(dir_path, labels_file)
    train_file = path.join(dir_path, train_file)
    val_file = path.join(dir_path, val_file)
    test_file = path.join(dir_path, test_file)

    with open(labels_file, 'r') as fin:
        lines = fin.readlines()
        ids = [i for i in range(0, len(lines), same_images)]

    val_num = int(len(ids) * config.VALIDATION_PERCENT)
    test_num = int(len(ids) * config.TESTING_PERCENT)
    train_num = len(ids) - val_num - test_num

    train = np.random.choice(ids, size=train_num, replace=False)
    ids = np.setdiff1d(ids, train)

    val = np.random.choice(ids, size=val_num, replace=False)

    with open(labels_file, 'r') as fin,      \
            open(train_file, 'w') as ftrain, \
            open(val_file, 'w') as fval,     \
            open(test_file, 'w') as ftest:
        lines = fin.readlines()

        for i in range(0, len(lines), same_images):
            if i in train:
                fresult = ftrain
            elif i in val:
                fresult = fval
            else:
                fresult = ftest

            for off in range(same_images):
                print(lines[i+off], end='', file=fresult)
    

if __name__ == '__main__':
    normalizer = DataNormalizer(dir_path=config.DATA_PATH, 
                                raw_labels_file=config.RAW_LABELS_FILE,
                                labels_file=config.LABELS_FILE, 
                                image_height=config.IMAGE_HEIGHT,
                                image_width=config.IMAGE_WIDTH)
    chars_used = normalizer()
    print(chars_used)

    divide_dataset(dir_path=config.DATA_PATH, 
                   labels_file=config.LABELS_FILE, 
                   train_file=config.TRAIN_FILE, 
                   val_file=config.VAL_FILE, 
                   test_file=config.TEST_FILE,
                   same_images=normalizer.images_number)
