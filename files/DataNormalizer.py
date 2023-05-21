import os
from  torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision.transforms import Resize
import torch

import Config


class DataNormalizer:
    """
        Extract labels from words.txt, delete invalid images and create new dataset with normalized pictures
    """

    def __init__(self, dir_path: str, raw_labels_file: str, labels_file: str, image_height: int, image_width: int):
        """
            Args:
                dir_path: the directory, where labels and images have been saved
                raw_labels_file: the file with the information about images
        """
        self.dir_path = dir_path
        self.raw_labels_file = raw_labels_file
        self.labels_file = labels_file

        self.image_height = image_height
        self.image_width = image_width

    def __call__(self):

        raw_labels_file = os.path.join(self.dir_path, self.raw_labels_file)
        labels_file = os.path.join(self.dir_path, self.labels_file)

        dictionary = dict()
        cur_label_id = 1

        with open(raw_labels_file, 'r') as raw_labels, open(labels_file, 'w') as labels:
            for line in raw_labels:
                if line[0] == '#':
                    continue

                line = line.split()

                image_path_parts = line[0].split('-')
                image_path = os.path.join(
                    self.dir_path, 'words', image_path_parts[0], '-'.join(image_path_parts[:2]), line[0] + '.png')

                if not os.path.isfile(image_path):
                    continue

                label = ' '.join(line[8:])

                if os.path.getsize(image_path)==0 or line[1] != "ok" or label=='.' or label=='-':
                    os.remove(image_path)
                    continue

                label = self.make_valid_label(label, Config.MaxLabelLength)

                image = read_image(
                    image_path, ImageReadMode.GRAY).to(torch.float)
                
                image = preprocess_image(image, self.image_height, self.image_width)
                image_mean = torch.mean(image).item()
                image_std = torch.std(image).item()

                save_image(
                    image, image_path, normalize=True)
                
                print(image_path, image_mean, image_std, label, file=labels)

                for symbol in list(label):
                    if symbol not in dictionary:
                        dictionary[symbol] = cur_label_id
                        cur_label_id += 1

            return dictionary

    def make_valid_label(self, label : str, max_len : int) -> str:
        """
            define the length of the label in ctc terms
        """
        cur_len = 1

        for i in range(1, len(label)):
            if label[i] == label[i - 1]:
                # adding 2, cause between same symbols there should be a special blank
                cur_len += 2
            else:
                cur_len += 1

            if cur_len > max_len:
                return label[:i]

        return label

def preprocess_image(image: torch.Tensor, image_height, image_width, position='left') -> torch.Tensor:
    """
        image: np.ndarray, that keep the image required to be transformed

        position: place, where to store the image in the resulting matrix
                       could be 'left', 'right' or 'center'.
    """

    (cur_image_height, cur_image_width) = image.shape[1:]

    if cur_image_height == image_height and cur_image_width == image_width:
        return image

    height_rel = cur_image_height / image_height
    width_rel = cur_image_width / image_width

    if height_rel > width_rel:
        new_image_height = image_height
        new_image_width = int(cur_image_width / height_rel)
    else:
        new_image_height = int(cur_image_height / width_rel)
        new_image_width = image_width

    transformer = Resize(size=(new_image_height, new_image_width), antialias=None)
    image = transformer(image)

    req_image = torch.full(
        (1, image_height, image_width), torch.max(image), dtype=torch.float)

    if position == 'left':
        req_image[:, :new_image_height, :new_image_width] = image
    elif position == 'right':
        req_image[:, -new_image_height:, -new_image_width:] = image

    return req_image


if __name__ == '__main__':
    normalizer = DataNormalizer(Config.DataPath, Config.RawLabelsFile,
                                Config.LabelsFile, Config.ImageHeight,
                                Config.ImageWidth)
    chars_used = normalizer()
    print(chars_used)