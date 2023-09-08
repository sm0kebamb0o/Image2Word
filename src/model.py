import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
import os.path as path
from torch.utils.data import Dataset, DataLoader
import datetime
from tqdm import tqdm

import config


class Sample:
    """Stores sample for neural network."""
    def __init__(self, path, label, length):
        self.path = path
        self.label = label
        self.length = length


class CustomDataLoader(Dataset):
    """
    Provides access to the required data.
    Should be used only in Dataloader.
    """

    def __init__(self, dir_path: str, labels_file: str) -> None:
        self.samples = list()
        labels_file = path.join(dir_path, labels_file)

        with open(labels_file, 'r') as samples_file:
            for line in samples_file:
                line = line.split()

                label = ' '.join(line[1:])

                length = len(label)
                label_embedding = torch.zeros(
                    config.MAX_LABEL_LENGTH, dtype=torch.int32)
                for i, symbol in enumerate(label):
                    label_embedding[i] = config.TERMINALS_TO_INDEXES[symbol]

                self.samples.append(Sample(line[0], label_embedding, length))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ind: int) -> tuple:
        sample = self.samples[ind]
        image = read_image(sample.path, ImageReadMode.GRAY).to(torch.float)
        return image, sample.label, sample.length

class SimpleHTR(nn.Module):
    """Simple Handwriteen Text Recognition System."""
    def __init__(self) -> None:
        """
        Keyword arguments:
        parameters_file: file, where all weights are stored
        device: device on which all computations should be done
        """
        super(SimpleHTR, self).__init__()
        self.normalization = nn.LayerNorm(normalized_shape=[1,config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
        self.__setCNN()
        self.__setRNN()

    def __setCNN(self):
        CHANNELS_NUMBER = [1,32,64,128,128,256]
        CONV_KERNEL_SIZES = [5,5,3,3,3]
        POOL_KERNEL_SIZS = POOL_STRIDES = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]

        layers = []
        for i in range(5):
            layers.append(nn.Conv2d(in_channels=CHANNELS_NUMBER[i],
                                    out_channels=CHANNELS_NUMBER[i+1],
                                    kernel_size=CONV_KERNEL_SIZES[i],
                                    padding='same'))
            if i & 1:
                layers.append(nn.BatchNorm2d(num_features=CHANNELS_NUMBER[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZS[i],
                                       stride=POOL_STRIDES[i], padding=0))
        self.layers = nn.ModuleList(layers)

    def __forwardCNN(self, x:torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def __setRNN(self):
        HiddenNum = 256
        self.rnn = nn.LSTM(input_size=HiddenNum, 
                           hidden_size=HiddenNum, 
                           num_layers=2, 
                           batch_first=True, 
                           bidirectional=True)
        # self.attention = nn.TransformerEncoderLayer(d_model=1024, nhead=4, batch_first=True, norm_first=True)
        self.filter = nn.Conv1d(in_channels=2*HiddenNum,
                                out_channels=config.TERMINALS_NUMBER+1,
                                kernel_size=3,
                                padding='same')
        # self.logits = nn.Linear(in_features=1024*32, out_features=(config.TERMINALS_NUMBER+1)*32)
        self.softmax = nn.LogSoftmax(dim=1)

    def __forwardRNN(self, x:torch.Tensor):
        x = x.squeeze(dim=2).transpose(1, 2)
 
        x, (_, _) = self.rnn(x)
        # x = self.attention(x)

        x = x.transpose(1, 2)
        # batch_size, timestaps, _ = x.shape
        # x = x.reshape(batch_size,-1)
        x = self.filter(x)
        # x = x.reshape(batch_size, -1, timestaps)
        return self.softmax(x)

    def forward(self, images:torch.Tensor) -> torch.Tensor:
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
        return images


class ModelHandler:
    """
    Special class for more convenient use of NN
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dir_path: str,
                 params_file='BestParams.pth',
                 cur_params_file='TrainParams.pth',
                 device=torch.device('cpu')):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.best_params_file = path.join(dir_path, params_file)
        self.cur_params_file = path.join(dir_path, cur_params_file)
        self.max_epoch = 0
        self.min_loss = float('inf')
        self.device = device
        self.history = [(self.min_loss, self.min_loss)]

    def recover(self, train=False):
        if not train:
            if not path.isfile(self.best_params_file):
                raise "Trying to evaluate NN before training it"
            self.model.load_state_dict(torch.load(self.best_params_file))
        else:
            if not path.isfile(self.cur_params_file):
                return
            state = torch.load(self.cur_params_file)

            self.model.load_state_dict(state['model'])
            self.max_epoch = state['epoch']
            self.min_loss = state['loss']
            self.history = state['history']
            self.optimizer.load_state_dict(state['optimizer'])

    def save(self, epoch, best=False):
        if best:
            torch.save(self.model.state_dict(), self.best_params_file)
        else:
            state = {
                'epoch': epoch,
                'loss': self.min_loss,
                'history': self.history,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, self.cur_params_file)

    def get_parameters_number(self):
        self.recover(train=True)
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad), \
            sum(p.numel() for p in self.model.parameters())

    def get_training_epoch_number(self):
        self.recover(train=True)
        return self.max_epoch

    def get_min_loss(self):
        self.recover(train=True)
        return self.min_loss

    def get_stats(self):
        self.recover(train=True)
        return self.history


class Image2Word(nn.Module):
    """Handwriteen Text Recognition System."""

    def __init__(self) -> None:
        """
        Keyword arguments:
        parameters_file: file, where all weights are stored
        device: device on which all computations should be done
        """
        super(Image2Word, self).__init__()
        '''
        self.normalization = nn.LayerNorm(
            normalized_shape=[1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
        '''
        self.__setCNN()
        self.__setRNN()

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

    def __setRNN(self):
        HiddenNum = 512
        self.rnn = nn.LSTM(input_size=HiddenNum,
                           hidden_size=HiddenNum,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True)
        # self.attention = nn.TransformerEncoderLayer(d_model=1024, nhead=4, batch_first=True, norm_first=True)
        self.filter = nn.Conv1d(in_channels=2*HiddenNum,
                                out_channels=config.TERMINALS_NUMBER+1,
                                kernel_size=3,
                                padding='same')
        # self.logits = nn.Linear(in_features=1024*32, out_features=(config.TERMINALS_NUMBER+1)*32)
        self.softmax = nn.LogSoftmax(dim=1)

    def __forwardRNN(self, x: torch.Tensor):
        x = x.squeeze(dim=2).transpose(1, 2)

        x, (_, _) = self.rnn(x)
        # x = self.attention(x)

        x = x.transpose(1, 2)
        # batch_size, timestaps, _ = x.shape
        # x = x.reshape(batch_size,-1)
        x = self.filter(x)
        # x = x.reshape(batch_size, -1, timestaps)
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
        images = self.__forwardRNN(images)
        return self.softmax(images)
