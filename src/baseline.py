import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import sys
import argparse
import os.path as path
from torchmetrics.functional import char_error_rate

from preprocessing import WordPreprocessor, PositionMode
from utils import Sample, ModelHandler
import config


class BaseDataset(Dataset):
    """
    Provides access to the required data.
    Should be used only in Dataloader.
    """

    def __init__(self, labels_file: str) -> None:
        self.samples = list()

        self.preproccesor = WordPreprocessor(image_height=config.BASELINE.IMAGE_HEIGHT,
                                             image_width=config.BASELINE.IMAGE_WIDTH,
                                             position_mode=PositionMode.Left)

        code = np.vectorize(lambda key: config.TERMINALS_TO_INDEXES[key])

        with open(labels_file, 'r') as samples_file:
            for line in samples_file:

                line = line.split()

                label = ' '.join(line[1:])

                length = min(len(label), config.BASELINE.MAX_LABEL_LENGTH)
                label_code = torch.zeros(
                    config.BASELINE.MAX_LABEL_LENGTH, dtype=torch.int32)
                label_code[:length] = torch.Tensor(code(list(label))[:length])

                self.samples.append(Sample(line[0], label_code, length))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, ind: int) -> tuple:
        sample = self.samples[ind]
        image = cv.imread(sample.path, cv.IMREAD_GRAYSCALE)
        image_preprocessed = self.preproccesor(image)[0]
        image_preprocessed = torch.FloatTensor(image_preprocessed)
        return image_preprocessed.unsqueeze(dim=0), sample.label, sample.length


class BaseModel(nn.Module):
    """Simple Handwriteen Text Recognition System."""

    def __init__(self) -> None:
        """
        Keyword arguments:
        parameters_file: file, where all weights are stored
        device: device on which all computations should be done
        """
        super(BaseModel, self).__init__()
        self.normalization = nn.LayerNorm(normalized_shape=[1, 
                                                            config.BASELINE.IMAGE_HEIGHT, 
                                                            config.BASELINE.IMAGE_WIDTH])
        self.__setCNN()
        self.__setRNN(256, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def __setCNN(self):
        CHANNELS_NUMBER = [1, 32, 64, 128, 128, 256]
        CONV_KERNEL_SIZES = [5, 5, 3, 3, 3]
        POOL_KERNEL_SIZS = POOL_STRIDES = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]

        layers = []
        for i in range(len(CONV_KERNEL_SIZES)):
            layers.append(nn.Conv2d(in_channels=CHANNELS_NUMBER[i],
                                    out_channels=CHANNELS_NUMBER[i+1],
                                    kernel_size=CONV_KERNEL_SIZES[i],
                                    padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZS[i],
                                       stride=POOL_STRIDES[i], padding=0))
        self.layers = nn.ModuleList(layers)

    def __forwardCNN(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def __setRNN(self, hidden_size, num_layers):
        self.rnn = nn.LSTM(input_size=256,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True)
        self.filter = nn.Conv1d(in_channels=2*hidden_size,
                                out_channels=config.TERMINALS_NUMBER+1,
                                kernel_size=1,
                                padding='same')

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


def make_initial_setup():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        data_loader_args = {'num_workers': 1, 'pin_memory': True}
    else:
        device = torch.device('cpu')
        data_loader_args = {}

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    return device, data_loader_args


def create_data_loaders(
        train_file: str,
        val_file: str,
        test_file: str,
        batch_size: int,
        data_loader_args: dict):
    train_dataset = BaseDataset(train_file)
    val_dataset = BaseDataset(val_file)
    test_dataset = BaseDataset(test_file)

    print('Training Dataset size:', len(train_dataset))
    print('Validation Dataset size:', len(val_dataset))
    print('Testing Dataset size:', len(test_dataset))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              **data_loader_args)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            **data_loader_args)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             **data_loader_args)
    return train_loader, val_loader, test_loader


def train(handler: ModelHandler,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          epoch_n: int,
          loss_criteria,
          log_file,
          show_status_bar: bool):
    handler.recover(train=True)
    prev_loss = [handler.history[-1]]
    epoch_bar_format = 'Training: {percentage:3.0f}%|{bar:25}| Epoch {n_fmt}/{total_fmt}, Train Loss {postfix[0][0]}, Validation Loss {postfix[0][1]}'
    batch_bar_format = '{desc}: {percentage:3.0f}%|{bar:25}| Batch {n_fmt}/{total_fmt}, Remaining time {remaining}'
    for epoch in tqdm(range(handler.max_epoch+1, epoch_n+1),
                      bar_format=epoch_bar_format,
                      postfix=prev_loss,
                      disable=show_status_bar,
                      initial=handler.max_epoch,
                      total=epoch_n):
        mean_train_loss = 0.
        train_batches = 0
        epoch_train_start = datetime.datetime.now()
        handler.model.train()
        for images, labels, lengths in tqdm(train_dataloader,
                                            leave=False,
                                            desc='  Gradient descent',
                                            bar_format=batch_bar_format,
                                            disable=show_status_bar):
            images, labels, lengths = images.to(
                handler.device), labels.to(handler.device), lengths.to(handler.device)

            handler.optimizer.zero_grad()
            images_transformed = handler.model.forward(images)

            input_lengths = torch.full(size=[images_transformed.shape[0]],
                                       fill_value=config.BASELINE.MAX_LABEL_LENGTH,
                                       dtype=torch.int32,
                                       device=handler.device)

            symbols_probabilities = torch.permute(
                images_transformed, (2, 0, 1))

            loss = loss_criteria(symbols_probabilities,
                                 labels,
                                 input_lengths,
                                 lengths)

            loss_val = loss.item()

            loss.backward()
            handler.optimizer.step()

            mean_train_loss += loss_val
            train_batches += 1

        mean_train_loss /= train_batches

        print(f'Epoch: {epoch} [Training], \
              {(datetime.datetime.now() - epoch_train_start).total_seconds():0.2f} s', file=log_file)
        print('Mean error value during training',
              mean_train_loss, file=log_file)
        print(file=log_file)

        epoch_val_start = datetime.datetime.now()
        mean_val_loss, _ = evaluate(handler,
                                    val_dataloader,
                                    loss_criteria,
                                    description='  Validation',
                                    disable=show_status_bar,
                                    leave=False)

        print(f'Epoch: {epoch} [Validation], \
              { (datetime.datetime.now() - epoch_val_start).total_seconds():0.2f} s', file=log_file)
        print('Mean error value during validation',
              mean_val_loss, file=log_file)
        print(file=log_file)

        handler.history.append((mean_train_loss, mean_val_loss))
        prev_loss[0] = handler.history[-1]
        if mean_val_loss < handler.min_loss:
            handler.min_loss = mean_val_loss
            handler.save(epoch=epoch, best=True)
        handler.save(epoch=epoch)
    handler.max_epoch = epoch_n


def evaluate(handler: ModelHandler,
             test_dataloader: DataLoader,
             loss_criteria,
             description,
             cer=False,
             disable=False,
             leave=True):
    handler.model.eval()
    mean_test_loss = 0.
    test_batches = 0

    mean_acc = 0.

    bar_format = '{desc}: {percentage:3.0f}%|{bar:25}| Batch {n_fmt}/{total_fmt}, Remaining time {remaining}'
    with torch.no_grad():
        for images, labels, lengths in tqdm(test_dataloader,
                                            bar_format=bar_format,
                                            desc=description,
                                            disable=disable,
                                            leave=leave):
            images, labels, lengths = images.to(
                handler.device), labels.to(handler.device), lengths.to(handler.device)
            images_transformed = handler.model.forward(images)
            input_lengths = torch.full(
                size=[images_transformed.shape[0]],
                fill_value=config.BASELINE.MAX_LABEL_LENGTH,
                dtype=torch.int32,
                device=handler.device)

            symbols_probabilities = torch.permute(
                images_transformed, (2, 0, 1))

            loss = loss_criteria(symbols_probabilities,
                                 labels,
                                 input_lengths,
                                 lengths)
            mean_test_loss += loss.item()
            test_batches += 1

            if cer:
                symbols_probabilities = symbols_probabilities.permute(1, 2, 0)
                mean_acc += calculate_cer(symbols_probabilities, labels)
    mean_test_loss /= test_batches
    mean_acc /= test_batches

    return mean_test_loss, mean_acc


def convert_to_word(word_embedding: torch.Tensor) -> str:
    word = ''
    for val in word_embedding:
        word += config.INDEXES_TO_TERMINALS[val]
    return word


def best_path(symbols_probability: torch.Tensor) -> torch.Tensor:
    """
    Implements best path decoding method to the probabilities matrix.

    Keyword arguments:
    symbols_probability: matrix of size CxT, where C - capacity of multiple terminals,
                         T - number of time-staps
                
    Return value:
    A recognized word coded with the config.MapTable
    """
    most_probable_symbols = torch.argmax(symbols_probability, dim=1)
    most_probable_label = torch.unique_consecutive(
        most_probable_symbols, dim=1)

    for i, label in enumerate(most_probable_label):
        mask = label != 0
        most_probable_label[i, :mask.sum()] = label[mask]
        most_probable_label[i, mask.sum():] = 0
    return most_probable_label

def calculate_cer(symbols_probabilities:torch.Tensor,
                  labels:torch.Tensor) -> float:
    predicts = best_path(symbols_probabilities)

    def decode(predict): return convert_to_word(predict)

    predicts_str = list(map(decode, predicts))
    labels_str = list(map(decode, labels))

    return char_error_rate(predicts_str, labels_str).item()


def create_parser():
    parser = argparse.ArgumentParser(description='Image2Vec',
                                     epilog='(c) Mamedov Timur 2023. Moscow State University.')
    subparsers = parser.add_subparsers(dest='mode')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs',
                              type=int,
                              help='The required number of epochs in training')
    train_parser.add_argument('--show',
                              action='store_true',
                              default=False,
                              help='Provides status bar')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('file_path',
                                help='The relative path to the required image')
    '''
    predict_parser.add_argument('-n',
                             '--new',
                             action='store_true',
                             default=False,
                             help='Flag for your own image')
    '''
    predict_parser.add_argument('-b',
                                '--beam_width',
                                type=int,
                                default=25,
                                help='The required beam width in decoding',
                                metavar='VALUE')
    predict_parser.add_argument('-lm',
                                '--lm_influence',
                                type=float,
                                default=config.LM_INFLUENCE,
                                help='The required language model influence in decoding',
                                metavar='VALUE')
    predict_parser.add_argument('--show',
                                action='store_true',
                                default=False,
                                help='Showes transformed images')
    _ = subparsers.add_parser('stats')
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--show',
                             action='store_true',
                             default=False,
                             help='Provides status bar')
    return parser


################### Main ###################

if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    device, data_loader_args = make_initial_setup()

    model = BaseModel()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.BASELINE.LR)
    handler = ModelHandler(model,
                           optimizer,
                           params_file=path.join(
                               config.PARAMS, config.BASELINE.BEST_PARAMS),
                           cur_params_file=path.join(
                               config.PARAMS, config.BASELINE.PARAMS),
                           device=device)

    if namespace.mode == 'train':
        handler.recover(train=True)
        loss_function = nn.CTCLoss(blank=config.BLANK,
                                   reduction='mean')

        train_loader, val_loader, _ = create_data_loaders(train_file=path.join(
            config.DATA_PATH, config.TRAIN_FILE),
            val_file=path.join(
            config.DATA_PATH, config.VAL_FILE),
            test_file=path.join(
            config.DATA_PATH, config.TEST_FILE),
            batch_size=config.BATCH_SIZE,
            data_loader_args=data_loader_args)
        with open(config.BASELINE.LOG_FILE, 'a') as log_file:
            train(handler=handler,
                  train_dataloader=train_loader,
                  val_dataloader=val_loader,
                  epoch_n=namespace.epochs,
                  loss_criteria=loss_function,
                  log_file=log_file,
                  show_status_bar=not namespace.show)

    elif namespace.mode == 'eval':
        handler.recover()
        loss_function = nn.CTCLoss(blank=config.BLANK,
                                   reduction='mean')

        _, _, test_loader = create_data_loaders(train_file=path.join(
            config.DATA_PATH, config.TRAIN_FILE),
            val_file=path.join(
            config.DATA_PATH, config.VAL_FILE),
            test_file=path.join(
            config.DATA_PATH, config.TEST_FILE),
            batch_size=config.BATCH_SIZE,
            data_loader_args=data_loader_args)
        loss, cer = evaluate(handler=handler,
                             test_dataloader=test_loader,
                             loss_criteria=loss_function,
                             description='Evaluation',
                             disable=not namespace.show,
                             cer=True)
        print(f"Loss on test dataset is {loss}.")
        print(f"CER on test dataset is {cer}.")
    elif namespace.mode == 'predict':
        if not path.isfile(namespace.file_path):
            raise RuntimeError("No such file")

        handler.recover()

        start = datetime.datetime.now()
        image = cv.imread(namespace.file_path, cv.IMREAD_GRAYSCALE)
        preprocessor = WordPreprocessor(
            config.BASELINE.IMAGE_HEIGHT, config.BASELINE.IMAGE_WIDTH, PositionMode.Left)
        image_modes = preprocessor(image)[0]

        sample = torch.FloatTensor(image_modes).view(size=[1,1, *image_modes.shape]).to(device)

        handler.model.eval()
        with torch.no_grad():
            images_predicted = handler.model.forward(sample)
        elapsed_time = (datetime.datetime.now() - start).total_seconds()

        label = convert_to_word(best_path(images_predicted).squeeze())
        print(label)

        if namespace.show:
            print(f"Elapsed time {elapsed_time}.")
            fig, axs = plt.subplots(nrows=image_modes.shape[0], ncols=1)
            for i in range(image_modes.shape[0]):
                axs[i].imshow(image_modes[i], cmap='gray')
            fig.text(0.0, 0.0, str(datetime.datetime.now()),
                     fontsize=10, color='gray', alpha=0.6)
            fig.suptitle(label, fontweight='heavy', fontsize='x-large')
            fig.tight_layout()
            plt.show()

    elif namespace.mode == 'stats':
        handler.recover(train=True)
        print(
            f"Number of trainable parameters {handler.get_parameters_number()[0]}.")

        training, testing = list(zip(*handler.get_stats()))

        plt.plot(training, label='Traininig Dataset', color='#8205f0')
        plt.plot(testing, label='Validation Dataset', color='#f55a00')
        plt.title('Loss value during training')
        plt.xlabel('Epochs number')
        plt.ylabel('Loss value')
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-')
        plt.grid(which='minor', linestyle=':')
        plt.tight_layout()
        plt.legend()
        plt.text(
            0.0, 0.0, str(datetime.datetime.now()),
            fontsize=10, color='gray', alpha=0.6
        )
        plt.show()
    else:
        print("Check a list of possible arguments.")
