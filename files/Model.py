import torch
from torchvision.io import read_image, ImageReadMode
from os.path import join
from torch.utils.data import Dataset
from enum import Enum

import Config

class Sample:
    """
        Store samples for neural network
    """
    def __init__(self, path, label, length):
        self.path = path
        self.label = label
        self.length = length

class CustomDataLoader(Dataset):
    """
        Custom data loader that provides access to the required data.
    """
    def __init__(self, dir_path, labels_file):
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
                    Config.MaxLabelLength, dtype=torch.long)
                for i, symbol in enumerate(label):
                    label_embedding[i] = Config.MapTable[symbol]

                self.samples.append(Sample(line[0], label_embedding, length))
                self.image_mean.append(float(line[1]))
                self.image_std.append(float(line[2]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, ind):
        sample = self.samples[ind]
        image = read_image(join('..', sample.path), ImageReadMode.GRAY).to(torch.float)

        image = image - self.image_mean[ind]
        image = (image / self.image_std[ind]) if self.image_std[ind] > 0 else image

        return image, sample.label, sample.length


class SimpleHTR(torch.nn.Module):
    def __init__(self, parameters_file, device=torch.device('cpu')):
        super(SimpleHTR, self).__init__()
        self.used_device = device
        self.parameters_file = parameters_file
        self.setCNN()
        self.setRNN()

    def setCNN(self):

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


    def forwardCNN(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


    def setRNN(self):
        HiddenNum = 256
        self.rnn = torch.nn.LSTM(
            input_size=HiddenNum, hidden_size=HiddenNum, num_layers=2, batch_first=True, bidirectional=True)
        
        self.filter = torch.nn.init.trunc_normal_(torch.empty(
            (Config.CharNum + 1, 2 * HiddenNum, 1, 1)), std=0.1).to(self.used_device)

    def forwardRNN(self, x):
        x = x.squeeze(dim=2).transpose(1, -1)
 
        x, (_, _) = self.rnn(x)

        x = x.transpose(1, 2).unsqueeze(dim=-1)
        x = torch.nn.functional.conv2d(x, self.filter, padding='same').squeeze(dim=-1)
        return x

    def forward(self, images):
        images = self.forwardCNN(images)
        images = self.forwardRNN(images)
        return images
    
    def load_previous_state(self, state : dict):
        self.load_state_dict(state['state_dict'])
        self.filter = state['filter']
    
    def save(self, step, epoch, optimizer_state_dict):
        state = {
            'epoch': epoch,
            'step':step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer_state_dict,
            'filter': self.filter,
        }
        torch.save(state, self.parameters_file)
