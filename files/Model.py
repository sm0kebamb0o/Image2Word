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
    def __init__(self, dir_path:str, labels_file:str) -> None:
        self.samples = list()
        self.image_mean = list()
        self.image_std = list()
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

    def __getitem__(self, ind:int) -> tuple:
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
    def __init__(self, 
                 model:nn.Module, 
                 optimizer:torch.optim.Optimizer,
                 dir_path:str,
                 params_file='BestParams.pth',
                 cur_params_file='TrainParams.pth',
                 device=torch.device('cpu')):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.params_file = path.join(dir_path, params_file)
        self.cur_params_file = path.join(dir_path, cur_params_file)
        self.max_epoch = 0
        self.min_loss = float('inf')
        self.recovered_test = False
        self.recovered_train = False
        self.device = device
        self.history = []

    def recover(self, train=False):
        if not train:
            self.recovered_test = True
            self.recovered_train = False
            file = self.params_file
        else:
            self.recovered_train = True
            self.recovered_test = False
            file = self.cur_params_file
        if not path.isfile(file):
            return
        state = torch.load(file)

        self.model.load_state_dict(state['model'])
        self.max_epoch = state['epoch']
        self.min_loss = state['loss']
        self.history = state['history']
        self.optimizer.load_state_dict(state['optimizer'])

    def save(self, epoch, best=False):
        if best:
            file = self.params_file
        else:
            file = self.cur_params_file
        
        state = {
            'epoch': epoch,
            'loss':self.min_loss,
            'history':self.history,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, file)
    
    def get_parameters_number(self):
        if not (self.recovered_train or self.recovered_test):
            raise "Incorrect execution order"
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad), \
               sum(p.numel() for p in self.model.parameters())
    
    def get_training_epoch_number(self):
        if not (self.recovered_train or self.recovered_test):
            raise "Incorrect execution order"
        return self.max_epoch
    
    def get_min_loss(self):
        if not (self.recovered_train or self.recovered_test):
            raise "Incorrect execution order"
        return self.min_loss
    
    def get_stats(self):
        if not (self.recovered_train or self.recovered_test):
            raise "Incorrect execution order"
        return self.history
    
    def train(self, 
              train_dataloader: DataLoader, 
              val_dataloader:DataLoader, 
              epoch_n:int, 
              loss_criteria):
        if not self.recovered_train:
            self.recover(train=True)
        bar_format = 'Training: {percentage:3.0f}%|{bar:25}| Epoch {n_fmt}/{total_fmt}, Remainig time {remaining}'
        for epoch in tqdm(range(self.max_epoch+1, epoch_n+1), bar_format=bar_format):
            mean_train_loss = 0.
            train_batches = 0
            epoch_train_start = datetime.datetime.now()
            self.model.train()
            for images, labels, lengths in train_dataloader:
                images, labels, lengths = images.to(
                    self.device), labels.to(self.device), lengths.to(self.device)

                self.optimizer.zero_grad()
                images_transformed = self.model.forward(images)

                input_lengths = torch.full(
                    size=[images_transformed.shape[0]],
                    fill_value=config.MAX_LABEL_LENGTH,
                    dtype=torch.int32, 
                    device=self.device)

                symbols_probabilities = torch.permute(
                    images_transformed, (2, 0, 1))

                loss = loss_criteria(symbols_probabilities, 
                                     labels,
                                     input_lengths, 
                                     lengths)

                loss_val = loss.item()

                loss.backward()
                self.optimizer.step()

                mean_train_loss += loss_val
                train_batches += 1

            mean_train_loss /= train_batches
            print(f'Эпоха: {epoch} [Обучение], \
                  {(datetime.datetime.now() - epoch_train_start).total_seconds():0.2f} сек')
            print('Среднее значение функции потерь на обучении', mean_train_loss)

            epoch_val_start = datetime.datetime.now()
            mean_val_loss = self.evaluate(val_dataloader,
                                          loss_criteria,
                                          desc='',
                                          disable=True)
            print(f'Эпоха: {epoch} [Валидация], \
                  { (datetime.datetime.now() - epoch_val_start).total_seconds():0.2f} сек')
            print('Среднее значение функции потерь на валидации', mean_val_loss)

            self.history.append((mean_train_loss, mean_val_loss))
            if mean_val_loss < self.min_loss:
                self.min_loss = mean_val_loss
                self.save(epoch=epoch, best=True)
            self.save(epoch=epoch)
        self.max_epoch = epoch_n
    
    def evaluate(self, test_dataloader, loss_criteria, desc, disable=False):
        self.model.eval()
        mean_test_loss = 0.
        test_batches = 0
        bar_format = '{desc}: {percentage:3.0f}%|{bar:25}| Batch {n_fmt}/{total_fmt}'
        with torch.no_grad():
            for images, labels, lengths in tqdm(test_dataloader, 
                                                bar_format=bar_format,
                                                desc=desc,
                                                disable=disable):
                images, labels, lengths = images.to(
                    self.device), labels.to(self.device), lengths.to(self.device)
                images_transformed = self.model.forward(images)
                input_lengths = torch.full(
                    size=[images_transformed.shape[0]],
                    fill_value=config.MAX_LABEL_LENGTH,
                    dtype=torch.int32,
                    device=self.device)

                symbols_probabilities = torch.permute(
                    images_transformed, (2, 0, 1))

                loss = loss_criteria(symbols_probabilities,
                                     labels,
                                     input_lengths,
                                     lengths)
                mean_test_loss += loss.item()
                test_batches += 1
        mean_test_loss /= test_batches
        return mean_test_loss

class Image2Word(nn.Module):
    """Handwriteen Text Recognition System."""

    def __init__(self) -> None:
        """
        Keyword arguments:
        parameters_file: file, where all weights are stored
        device: device on which all computations should be done
        """
        super(Image2Word, self).__init__()
        self.normalization = nn.LayerNorm(
            normalized_shape=[1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
        self.__setCNN()
        self.__setRNN()

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
        return self.softmax(x)

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
        return images
