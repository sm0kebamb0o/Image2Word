import numpy as np
from enum import Enum
import cv2 as cv

import utils
from deslant_img import deslant_img

class PositionMode(Enum):
    """Stores diferent positioning modes."""
    Left = 0
    Right = 1
    Random = 2

    def __len__(self) -> int:
        return super().__len__()


class WordProcessor:
    def __init__(self,
                 image_height: int,
                 image_width: int,
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

    def __call__(self, 
                 image: np.ndarray,
                 use_deslant:bool=False,
                 pick:int=0) -> list:
        """
        Remove noise from the image, remove slant and bring to the required view.
        Returns tuple of images with different level of noise and intensity.
        """
        assert len(image.shape) == 2
        images = self.__remove_noise(image)
        image = images[pick]
        if use_deslant:
            image = deslant_img(image)[0]
        return self.__resize_image(image)

    def __resize_image(self, image) -> np.ndarray:
        """
        Brings image to the required size and stores it at the specified 
        position in the resulting image.
        """
        (cur_image_height, cur_image_width) = image.shape

        if cur_image_height == self.image_height and cur_image_width == self.image_width:
            return 255-image

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
            shape=[self.image_height, self.image_width], fill_value=255, dtype=np.uint8)
        
        cur_position_mode = self.position_mode
        if self.position_mode == PositionMode.Random:
            cur_position_mode = np.random.choice([PositionMode.Left, PositionMode.Right], 1) 
        
        if cur_position_mode == PositionMode.Left:
            req_image[:new_image_height, :new_image_width] = image
        elif cur_position_mode == PositionMode.Right:
            req_image[-new_image_height:, -new_image_width:] = image

        return 255-req_image

    def __remove_noise(self, image: np.ndarray) -> tuple:
        """
        Remove all noise from the image. Returns 3 images with different 
        level of noise and intensity.

        Keyword arguments:
        image: image that should be tranformed
        """
        # Removing small noise by applying Gaussian blur
        blur_kernel = (utils.calc_gaussian_kernel_size(image.shape[0], 0.075),
                       utils.calc_gaussian_kernel_size(image.shape[1], 0.02))
        image_blur = cv.GaussianBlur(src=image,
                                     ksize=blur_kernel,
                                     sigmaX=0)

        # Binarizing image wiht Otsu's method
        _, image_bin = cv.threshold(src=image_blur,
                                    thresh=220,
                                    maxval=255,
                                    type=cv.THRESH_OTSU)

        # Putting image into bow with white edges for proper work
        temp_image = np.full(shape=[image_bin.shape[0]+2,
                                    image_bin.shape[1]+2], fill_value=255, dtype=np.uint8)
        temp_image[1:-1, 1:-1] = image_bin

        # Removing not necessary empty columns and rows
        def crop(axis):
            empty_axes = (temp_image == 255).all(axis=axis)
            _, elems, inds = utils.find_consecutive(empty_axes)
            left_bound = inds[elems == True][0]
            right_bound = inds[elems == False][-1]
            return left_bound, right_bound

        left_column, right_column = crop(axis=0)
        upper_row, lower_row = crop(axis=1)

        # Detecting strokes len in both directions for futher morphological operations
        lens, elems, _ = utils.find_consecutive(temp_image.ravel())
        h_stroke_len = int(np.median(lens[elems == 0]))

        lens, elems, _ = utils.find_consecutive(temp_image.T.ravel())
        v_stroke_len = int(np.median(lens[elems == 0]))

        # Actually cropping the image (^-^)
        image_bin = image_bin[upper_row:lower_row +
                              1, left_column:right_column+1]

        # Defining kernels sizes for morphological operations
        mid_kernel = (utils.calc_kernel_size(h_stroke_len, 0.25),
                      utils.calc_kernel_size(v_stroke_len, 0.25))
        small_kernel = (utils.calc_kernel_size(h_stroke_len, 0.2),
                        utils.calc_kernel_size(v_stroke_len, 0.2))

        image_border = cv.morphologyEx(src=image_bin,
                                       op=cv.MORPH_GRADIENT,
                                       kernel=cv.getStructuringElement(shape=cv.MORPH_RECT,
                                                                       ksize=mid_kernel))
        image_border += image_bin

        image_eroded = cv.morphologyEx(
            image_border, cv.MORPH_CLOSE, kernel=cv.getStructuringElement(cv.MORPH_RECT, small_kernel))

        image_eroded = cv.morphologyEx(image_eroded, cv.MORPH_ERODE,
                                       kernel=cv.getStructuringElement(cv.MORPH_RECT, small_kernel))
        return (image_bin, image_border, image_eroded)