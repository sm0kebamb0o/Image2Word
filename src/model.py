import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import random

import config
from processing import WordProcessor, PositionMode
from utils import Sample
from deslant_img import deslant_img


class CRNNDataset(Dataset):
    """
    Provides access to the required data.
    Should be used only in Dataloader.
    """

    def __init__(self, 
                 labels_file: str,
                 image_height:int,
                 image_width:int,
                 train:bool=False) -> None:
        self.samples = list()
        self.train = train

        self.proccesor = WordProcessor(image_height=image_height,
                                       image_width=image_width,
                                       position_mode=PositionMode.Random)

        code = np.vectorize(lambda key: config.TERMINALS_TO_INDEXES[key])

        with open(labels_file, 'r') as samples_file:
            for line in samples_file:

                line = line.split()

                label = ' '.join(line[1:])

                length = len(label)
                label_code = torch.zeros(config.MAX_LABEL_LENGTH, dtype=torch.int32)
                label_code[:length] = torch.Tensor(code(list(label)))

                self.samples.append(Sample(line[0], label_code))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ind: int) -> tuple:
        sample : Sample = self.samples[ind]
        image = cv.imread(sample.path, cv.IMREAD_GRAYSCALE)
        if self.train:
            image_processed = self.proccesor(image,
                                             pick=random.randint(a=0, b=2),
                                             use_deslant=bool(random.randint(a=0, b=1)))
        else:
            image_processed = self.proccesor(image)
        image_pt = torch.FloatTensor(image_processed).unsqueeze(dim=0)
        return image_pt, sample.label
    

class CRNN(nn.Module):
    """Handwriteen Text Recognition System."""

    def __init__(self, 
                 lstm_hidden_size:int,
                 lstm_num_layers:int,
                 mlp_hidden_size:int,
                 dropout_p:float,
                 output_size:int) -> None:
        """
        Keyword arguments:
        parameters_file: file, where all weights are stored
        device: device on which all computations should be done
        """
        super(CRNN, self).__init__()
        self.normalization = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=True
        )
        self.__setCNN()
        self.__setRNN(
            hidden_size=lstm_hidden_size, 
            num_layers=lstm_num_layers,
            dropout_p=dropout_p,
            inner_size=mlp_hidden_size,
            output_size=output_size
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def __setCNN(self):
        CHANNELS_NUMBER = [1, 64, 128, 128, 256, 256, 512, 512]
        CONV_KERNEL_SIZES = [5, 5, 3, 3, 3, 3, 3]
        POOL_KERNEL_SIZS = POOL_STRIDES = [(2, 2), (2, 1), (2, 2), (), (2, 2), (2, 1), (2, 1)]

        layers = []
        for i in range(len(CONV_KERNEL_SIZES)):
            layers.append(nn.Conv2d(in_channels=CHANNELS_NUMBER[i],
                                    out_channels=CHANNELS_NUMBER[i+1],
                                    kernel_size=CONV_KERNEL_SIZES[i],
                                    padding='same'))
            layers.append(nn.BatchNorm2d(
                num_features=CHANNELS_NUMBER[i+1]))
            layers.append(nn.ReLU())
            if i % 4 != 3:
                layers.append(nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZS[i],
                                           stride=POOL_STRIDES[i], padding=0))
        self.layers = nn.ModuleList(layers)

    def __forwardCNN(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def __setRNN(self, hidden_size, num_layers, dropout_p, inner_size, output_size):
        self.rnn = nn.LSTM(input_size=512,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True)
        self.filter = nn.Sequential(
            nn.Conv1d(in_channels=2*hidden_size,
                      out_channels=inner_size,
                      kernel_size=1,
                      padding='same'),
            nn.Dropout1d(p=dropout_p),
            nn.Conv1d(in_channels=inner_size,
                      out_channels=output_size,
                      kernel_size=1,
                      padding='same')
        )

    def __forwardRNN(self, x: torch.Tensor):
        x = x.squeeze(dim=2).transpose(1, 2)

        x, (_, _) = self.rnn(x)

        x = x.transpose(1, 2)
        x = self.filter(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies all layers to the passed batch.

        Keyword arguments:
        images: batch of images of size Bx1xHxW, where B - number of elements in batch,
                H - height of all images, W - width of all images in the batch
        
        Return value:
        Batch of the images of size BxCxT
        """
        images = self.normalization(images)
        images = self.__forwardCNN(images)
        images = self.__forwardRNN(images)
        return self.softmax(images)

'''
class Image2Word(nn.Module):
    """Handwriteen Text Recognition System."""

    def __init__(self) -> None:
        """
        Keyword arguments:
        parameters_file: file, where all weights are stored
        device: device on which all computations should be done
        """
        super(Image2Word, self).__init__()
        """
        self.normalization = nn.LayerNorm(
            normalized_shape=[1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
        """
        self.__setCNN()
        self.__setEncoder()
        self.filter = nn.Conv1d(in_channels=512,
                                out_channels=config.TERMINALS_NUMBER+1,
                                kernel_size=1,
                                padding='same')
        self.softmax = nn.LogSoftmax(dim=1)

    def __setCNN(self):
        CHANNELS_NUMBER = [1, 64, 128, 128, 256, 256, 512, 512]
        CONV_KERNEL_SIZES = [5, 5, 3, 3, 3, 3, 3]
        POOL_KERNEL_SIZS = POOL_STRIDES = [
            (2, 2), (2, 1), (2, 2), (), (2, 2), (2, 1), (2, 1)]

        layers = []
        for i in range(len(CONV_KERNEL_SIZES)):
            layers.append(nn.Conv2d(in_channels=CHANNELS_NUMBER[i],
                                    out_channels=CHANNELS_NUMBER[i+1],
                                    kernel_size=CONV_KERNEL_SIZES[i],
                                    padding='same'))
            if i % 3 == 2:
                layers.append(nn.BatchNorm2d(
                    num_features=CHANNELS_NUMBER[i+1]))
            layers.append(nn.ReLU())
            if i % 4 != 3:
                layers.append(nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZS[i],
                                           stride=POOL_STRIDES[i], padding=0))
        self.layers = nn.ModuleList(layers)

    def __forwardCNN(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __setEncoder(self, ):
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, 
                                                        nhead=4, 
                                                        batch_first=True, 
                                                        norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=4)
    
    def __forwardEncoder(self, x:torch.Tensor):
        x = x.squeeze(dim=2).transpose(1, 2)
        x = self.encoder(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies all layers to the passed batch.

        Keyword arguments:
        images: batch of images of size Bx1xHxW, where B - number of elements in batch,
                H - height of all images, W - width of all images in the batch
        
        Return value:
        Batch of the images of size BxCxT
        """
        # images = self.normalization(images)
        images = self.__forwardCNN(images)
        images = self.__forwardEncoder(images)
        images = images.transpose(1,2)
        images = self.filter(images)
        return self.softmax(images)
    

class PreprocessingDataset(Dataset):
    """
    Provides access to the required data.
    Should be used only in Dataloader.
    """

    def __init__(self, labels_file: str) -> None:
        self.samples = list()

        code = np.vectorize(lambda key: config.TERMINALS_TO_INDEXES[key])
        self.preproccesor = WordProcessor(image_height=config.IMAGE_HEIGHT,
                                             image_width=config.IMAGE_WIDTH,
                                             position_mode=PositionMode.Left)

        with open(labels_file, 'r') as samples_file:
            for line in samples_file:

                line = line.split()

                label = ' '.join(line[1:])

                length = len(label)
                label_code = torch.zeros(
                    config.MAX_LABEL_LENGTH, dtype=torch.int32)
                label_code[:length] = torch.Tensor(code(list(label)))

                self.samples.append(Sample(line[0], label_code, length))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ind: int) -> tuple:
        sample = self.samples[ind]
        image = cv.imread(sample.path, cv.IMREAD_GRAYSCALE)
        image_modes = self.preproccesor(image)
        image_modes = np.stack(image_modes, axis=0)
        images = torch.FloatTensor(image_modes)
        return images, sample.label, sample.length
    

class Image2Word_last(nn.Module):
    """Handwriteen Text Recognition System."""

    def __init__(self) -> None:
        """
        Keyword arguments:
        parameters_file: file, where all weights are stored
        device: device on which all computations should be done
        """
        super(Image2Word_last, self).__init__()
        self.__setCNN()
        self.__setEncoder()
        self.filter = nn.Conv1d(in_channels=512,
                                out_channels=config.TERMINALS_NUMBER+1,
                                kernel_size=1,
                                padding='same')
        self.softmax = nn.LogSoftmax(dim=1)

    def __setCNN(self):
        CHANNELS_NUMBER = [3, 64, 128, 128, 256, 256, 512, 512]
        CONV_KERNEL_SIZES = [5, 5, 3, 3, 3, 3, 3]
        POOL_KERNEL_SIZS = POOL_STRIDES = [
            (2, 2), (2, 1), (2, 2), (), (2, 2), (2, 1), (2, 1)]

        layers = []
        for i in range(len(CONV_KERNEL_SIZES)):
            layers.append(nn.Conv2d(in_channels=CHANNELS_NUMBER[i],
                                    out_channels=CHANNELS_NUMBER[i+1],
                                    kernel_size=CONV_KERNEL_SIZES[i],
                                    padding='same'))
            if i % 3 == 2:
                layers.append(nn.BatchNorm2d(
                    num_features=CHANNELS_NUMBER[i+1]))
            layers.append(nn.ReLU())
            if i % 4 != 3:
                layers.append(nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZS[i],
                                           stride=POOL_STRIDES[i], padding=0))
        self.layers = nn.ModuleList(layers)

    def __forwardCNN(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def __setEncoder(self):
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512,
                                                        nhead=4,
                                                        batch_first=True,
                                                        norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=2)

    def __forwardEncoder(self, x: torch.Tensor):
        x = x.squeeze(dim=2).transpose(1, 2)
        x = self.encoder(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies all layers to the passed batch.

        Keyword arguments:
        images: batch of images of size Bx1xHxW, where B - number of elements in batch,
                H - height of all images, W - width of all images in the batch
        
        Return value:
        Batch of the images of size BxCxT
        """
        # images = self.normalization(images)
        images = self.__forwardCNN(images)
        images = self.__forwardEncoder(images)
        images = images.transpose(1, 2)
        images = self.filter(images)
        return self.softmax(images)

class CustomDataLoader(Dataset):
    """
    Provides access to the required data.
    Should be used only in Dataloader.
    """

    def __init__(self, labels_file: str) -> None:
        self.samples = list()

        code = np.vectorize(lambda key: config.TERMINALS_TO_INDEXES[key])

        with open(labels_file, 'r') as samples_file:
            for line in samples_file:

                line = line.split()

                label = ' '.join(line[1:])

                length = len(label)
                label_code = torch.zeros(config.MAX_LABEL_LENGTH, dtype=torch.int32)
                label_code[:length] = torch.Tensor(code(list(label)))

                self.samples.append(Sample(line[0], label_code, length))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ind: int) -> tuple:
        sample = self.samples[ind]
        image = read_image(sample.path, ImageReadMode.GRAY).to(torch.float)
        return image, sample.label, sample.length
'''
