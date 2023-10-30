import torch
import numpy as np
import os.path as path
import torch.nn as nn


def find_consecutive(row):
    """
    Finds consecutive elements in row and returns subsequences lens, 
    elements and indexes of thie ending
    """
    assert len(row.shape) == 1
    # pairwise unequal (string safe)
    mask = row[1:] != row[:-1]
    # ищем где несовпадают значения соседних пикселей
    # must include last element posi
    inds = np.append(np.where(mask), row.shape[0] - 1)
    # находим разницу между правым и левым элементом, то есть нужные нам промежутки
    lens = np.diff(np.append(-1, inds))       # run lengths
    return (lens, row[inds], inds)


def calc_kernel_size(anchor: int, percent: float) -> int:
    """
    Calculate kernel size for OpenCV operations
    """
    kernel_size = int(percent * anchor)
    kernel_size += ((kernel_size & 1) + 1) & 1
    return kernel_size


def calc_gaussian_kernel_size(anchor: int, percent: float) -> int:
    """
    Calculate kernel size for OpenCV operations with Gaussian filter
    """
    kernel_size = calc_kernel_size(anchor, percent)
    return kernel_size if kernel_size > 1 else 3


class Sample:
    """Stores sample for neural network."""
    def __init__(self, path, label):
        self.path = path
        self.label = label


class ModelHandler:
    """
    Special class for more convenient use of NN
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 params_file='BestParams.pth',
                 cur_params_file='TrainParams.pth',
                 device=torch.device('cpu')):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler : torch.optim.lr_scheduler = lr_scheduler
        self.best_params_file = params_file
        self.cur_params_file = cur_params_file
        self.max_epoch = 0
        self.min_loss = float('inf')
        self.device = device
        self.history = [(self.min_loss, self.min_loss)]

    def recover(self, train=False):
        if not train:
            if not path.isfile(self.best_params_file):
                raise RuntimeError("Trying to evaluate NN before training it")
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
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state['lr_scheduler'])

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
                'lr_scheduler':self.lr_scheduler.state_dict()
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
