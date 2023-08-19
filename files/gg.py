from model import SimpleHTR, ModelHandler, CustomDataLoader, Image2Word
import config
from preprocessing import *
from decoding import best_path_decoding

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import sys


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
        val_file:str,
        test_file:str,
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

def create_parser():
    parser = argparse.ArgumentParser(description='Image2Vec',
                                     epilog='(c) Mamedov Timur 2023. Moscow State University.')
    subparsers = parser.add_subparsers(dest='mode')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('epochs',
                              type=int,
                              help='The required number of epochs in training')

    test_parser = subparsers.add_parser('predict')
    test_parser.add_argument('file_path',
                             help='The relative path to the required image')
    '''
    test_parser.add_argument('-n',
                             '--new',
                             action='store_true',
                             default=False,
                             help='Should be used when you are passing your own image')
    '''
    test_parser.add_argument('-b',
                             '--beam_width',
                             type=int,
                             default=25,
                             help='The required beam width in decoding',
                             metavar='VALUE')
    test_parser.add_argument('-lm',
                             '--lm_influence',
                             type=float,
                             default=config.LM_INFLUENCE,
                             help='The required language model influence in decoding',
                             metavar='VALUE')
    stats_parser = subparsers.add_parser('stats')

    return parser


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
    most_probable_label = torch.unique_consecutive(most_probable_symbols, dim=1)
    new_label = torch.zeros(
        size=(symbols_probability.shape[0], config.MAX_LABEL_LENGTH), dtype=torch.long)
    
    for i in range(most_probable_label.shape[0]):
        label = most_probable_label[i]
        mask = label != 0
        new_label[i, :mask.sum()] = label[mask]
    return new_label


if __name__=='__main__':
    '''
    device, data_loader_args = make_initial_setup()
    train_loader, val_loader, test_loader = create_data_loaders(data_path=config.DATA_PATH, 
                                                                train_file=config.TRAIN_FILE,
                                                                val_file=config.VAL_FILE,
                                                                test_file=config.TEST_FILE,
                                                                batch_size=config.BATCH_SIZE,
                                                                data_loader_args=data_loader_args)
    model = Image2Vec()
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=config.LEARNING_RATE)
    handler = ModelHandler(model, 
                           optimizer, 
                           dir_path=config.PARAMS_FOLDER, 
                           params_file='Best.pth', 
                           cur_params_file='Train.pth', 
                           device=device)
    handler.recover(train=True)
    loss_function = nn.CTCLoss(blank=0, 
                               reduction='mean')
    handler.train(train_dataloader=train_loader, 
                  val_dataloader=val_loader,
                  epoch_n=7,
                  loss_criteria=loss_function)
    '''
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    device, data_loader_args = make_initial_setup()
    model = Image2Word()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.LEARNING_RATE)
    handler = ModelHandler(model,
                           optimizer,
                           dir_path=config.PARAMS_FOLDER,
                           params_file='Best.pth',
                           cur_params_file='Train.pth',
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
        handler.train(train_dataloader=train_loader,
                      val_dataloader=val_loader,
                      epoch_n=namespace.epochs,
                      loss_criteria=loss_function)
    elif namespace.mode == 'predict':
        if not path.isfile(namespace.file_path):
            raise "No such file"
        
        handler.recover()
        image = cv.imread(namespace.file_path, cv.IMREAD_GRAYSCALE)
        preprocessor = DataPreprocessor(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, PositionMode.Left)
        images = preprocessor(image)
        images = np.stack(images, axis=0)

        images_predicted = handler.model.forward(torch.FloatTensor(images).unsqueeze(dim=1).to(device))

        for label in best_path(images_predicted):
            print(convert_to_word(label))
    elif namespace.mode == 'stats':
        handler.recover(train=True)
        print(f"Number of trainable parameters {handler.get_parameters_number()[0]}.")
        history = handler.get_stats()
        training = [epoch[0] for epoch in history]
        testing = [epoch[1] for epoch in history]

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
        