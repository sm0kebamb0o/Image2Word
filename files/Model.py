import torch
from torchvision.io import read_image, ImageReadMode
from os.path import join
from torch.utils.data import Dataset

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
        labels_file = join(dir_path, labels_file)

        with open(labels_file, 'r') as samples_file:
            for line in samples_file:
                line = line.split()

                label = ' '.join(line[3:])

                length = len(label)
                label_embedding = torch.zeros(
                    config.MAX_LABEL_LENGTH, dtype=torch.long)
                for i, symbol in enumerate(label):
                    label_embedding[i] = config.TERMINALS_TO_INDEXES[symbol]

                self.samples.append(Sample(line[0], label_embedding, length))
                self.image_mean.append(float(line[1]))
                self.image_std.append(float(line[2]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ind:int) -> tuple:
        sample = self.samples[ind]
        image = read_image(join('..', sample.path), ImageReadMode.GRAY).to(torch.float)

        image = image - self.image_mean[ind]
        image = (image / self.image_std[ind]) if self.image_std[ind] > 0 else image

        return image, sample.label, sample.length


class SimpleHTR(torch.nn.Module):
    """Simple Handwriteen Text Recognition System."""
    def __init__(self, 
                 parameters_file:str, 
                 device:torch.device=torch.device('cpu')) -> None:
        """
        Keyword arguments:
        parameters_file: file, where all weights are stored
        device: device on which all computations should be done
        """
        super(SimpleHTR, self).__init__()
        self.used_device = device
        self.parameters_file = parameters_file
        self.__setCNN()
        self.__setRNN()

    def __setCNN(self):

        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='same'),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
        
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
        
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0))
        
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0))
        
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
                                          torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0))


    def __forwardCNN(self, x:torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


    def __setRNN(self):
        HiddenNum = 256
        self.rnn = torch.nn.LSTM(
            input_size=HiddenNum, hidden_size=HiddenNum, num_layers=2, batch_first=True, bidirectional=True)
        
        self.filter = torch.nn.init.trunc_normal_(torch.empty(
            (config.TERMINALS_NUMBER + 1, 2 * HiddenNum, 1, 1)), std=0.1).to(self.used_device)

    def __forwardRNN(self, x:torch.Tensor):
        x = x.squeeze(dim=2).transpose(1, -1)
 
        x, (_, _) = self.rnn(x)

        x = x.transpose(1, 2).unsqueeze(dim=-1)
        x = torch.nn.functional.conv2d(x, self.filter, padding='same').squeeze(dim=-1)
        return x

    def forward(self, images:torch.Tensor) -> torch.Tensor:
        """
        Applies all layers to the passed batch.

        Keyword arguments:
        images: batch of images of size Bx1xHxW, where B - number of elements in batch,
                H - height of all images, W - width of all images in the batch
        
        Return value:
        Batch of the images of size BxCxT
        """
        images = self.__forwardCNN(images)
        images = self.__forwardRNN(images)
        return images
    
    def load_previous_state(self, state : dict) -> None:
        '''Loades stored weights.'''
        self.load_state_dict(state['state_dict'])
        self.filter = state['filter']
    
    def save(self, step:int, epoch:int, optimizer_state_dict:dict) -> None:
        '''Saves model's parameters.'''
        state = {
            'epoch': epoch,
            'step':step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer_state_dict,
            'filter': self.filter,
        }
        torch.save(state, self.parameters_file)
