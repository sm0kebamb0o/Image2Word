import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os.path as path
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import sys
import argparse

from model import ModelHandler, CustomDataLoader, Image2Word
import config
from preprocessing import WordPreprocessor, PositionMode

############ Required functions ############


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
        data_path: str,
        train_file: str,
        val_file: str,
        test_file: str,
        batch_size: int,
        data_loader_args: dict):
    train_dataset = CustomDataLoader(data_path, train_file)
    val_dataset = CustomDataLoader(data_path, val_file)
    test_dataset = CustomDataLoader(data_path, test_file)

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
    prev_loss = [handler.history[-1][0]]
    epoch_bar_format = 'Training: {percentage:3.0f}%|{bar:25}| Epoch {n_fmt}/{total_fmt}, Loss={postfix[0]}'
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
                                       fill_value=config.MAX_LABEL_LENGTH,
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
        prev_loss[0] = mean_train_loss

        print(f'Epoch: {epoch} [Training], \
              {(datetime.datetime.now() - epoch_train_start).total_seconds():0.2f} s', file=log_file)
        print('Mean error value during training',
              mean_train_loss, file=log_file)
        print(file=log_file)

        epoch_val_start = datetime.datetime.now()
        mean_val_loss = evaluate(handler,
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
        if mean_val_loss < handler.min_loss:
            handler.min_loss = mean_val_loss
            handler.save(epoch=epoch, best=True)
        handler.save(epoch=epoch)
    handler.max_epoch = epoch_n


def evaluate(handler: ModelHandler,
             test_dataloader: DataLoader,
             loss_criteria,
             description,
             disable=False,
             leave=True):
    handler.model.eval()
    mean_test_loss = 0.
    test_batches = 0
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
                fill_value=config.MAX_LABEL_LENGTH,
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
    mean_test_loss /= test_batches
    return mean_test_loss


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
    new_label = torch.zeros(
        size=(symbols_probability.shape[0], config.MAX_LABEL_LENGTH), dtype=torch.long)

    for i in range(most_probable_label.shape[0]):
        label = most_probable_label[i]
        mask = label != 0
        new_label[i, :mask.sum()] = label[mask]
    return new_label


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
    _ = subparsers.add_parser('stats')
    _ = subparsers.add_parser('eval')

    return parser


################### Main ###################

if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    device, data_loader_args = make_initial_setup()

    model = Image2Word()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.LEARNING_RATE)
    handler = ModelHandler(model,
                           optimizer,
                           dir_path=config.PARAMS_FOLDER,
                           params_file='Inv_Conv_Best.pth',
                           cur_params_file='Inv_Conv.pth',
                           device=device)

    if namespace.mode == 'train':
        handler.recover(train=True)
        loss_function = nn.CTCLoss(blank=0,
                                   reduction='mean')

        train_loader, val_loader, _ = create_data_loaders(data_path=config.DATA_PATH,
                                                          train_file=config.TRAIN_FILE,
                                                          val_file=config.VAL_FILE,
                                                          test_file=config.TEST_FILE,
                                                          batch_size=config.BATCH_SIZE,
                                                          data_loader_args=data_loader_args)
        with open(config.LOG_FILE, 'a') as log_file:
            train(handler=handler,
                  train_dataloader=train_loader,
                  val_dataloader=val_loader,
                  epoch_n=namespace.epochs,
                  loss_criteria=loss_function,
                  log_file=log_file,
                  show_status_bar=not namespace.show)

    elif namespace.mode == 'eval':
        handler.recover()
        loss_function = nn.CTCLoss(blank=0,
                                   reduction='mean')

        _, _, test_loader = create_data_loaders(data_path=config.DATA_PATH,
                                                train_file=config.TRAIN_FILE,
                                                val_file=config.VAL_FILE,
                                                test_file=config.TEST_FILE,
                                                batch_size=config.BATCH_SIZE,
                                                data_loader_args=data_loader_args)
        loss = evaluate(handler=handler,
                        test_dataloader=test_loader,
                        loss_criteria=loss_function,
                        description='Evaluation')
        print(f"Loss on test dataset is {loss}.")
    elif namespace.mode == 'predict':
        if not path.isfile(namespace.file_path):
            raise "No such file"

        handler.recover()
        image = cv.imread(namespace.file_path, cv.IMREAD_GRAYSCALE)
        preprocessor = WordPreprocessor(
            config.IMAGE_HEIGHT, config.IMAGE_WIDTH, PositionMode.Left)
        images = preprocessor(image)
        images = np.stack(images, axis=0)

        images_predicted = handler.model.forward(
            torch.FloatTensor(images).unsqueeze(dim=1).to(device))

        for label in best_path(images_predicted):
            print(convert_to_word(label))

    elif namespace.mode == 'stats':
        handler.recover(train=True)
        print(
            f"Number of trainable parameters {handler.get_parameters_number()[0]}.")

        training, testing = list(zip(*handler.get_stats()))

        import matplotlib.pyplot as plt
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
        plt.show()
    else:
        print("Check a list of possible arguments.")
