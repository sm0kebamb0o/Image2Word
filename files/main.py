import torch
import argparse
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import char_error_rate
from os.path import isfile, join
import sys
from multiprocessing import Process, Value

from model import CustomDataLoader, SimpleHTR
from decoding import *
import config
from language_model import LanguageModel
from data_normalizer import preprocess_image


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
        data_path:str, 
        labels_file:str, 
        testing_percent:float, 
        batch_size:int, 
        data_loader_args : dict):
    data = CustomDataLoader(data_path, labels_file)
    test_num = int(len(data) * testing_percent)
    train_num = len(data) - test_num
    train, test = random_split(data, [train_num, test_num])

    print('Training Dataset size:', len(train))
    print('Testing Dataset size:', len(test))

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, **data_loader_args)
    test_loader = DataLoader(test, batch_size=batch_size,
                             shuffle=False, **data_loader_args)
    return train_loader, test_loader


def convert_to_word(word_embedding:torch.Tensor)->str:
    word = ''
    for val in word_embedding:
        word += config.INDEXES_TO_TERMINALS[val]
    return word


def calculate_accuracy(symbols_probabilities: torch.Tensor,
                       labels: torch.Tensor,
                       accuracy: Value,
                       accuracy_cer: Value,
                       mode: DecodingMode,
                       beam_width: int = 10,
                       LM: LanguageModel = None):
    accuracy_value = 0.
    accuracy_cer_value = 0.
    if mode == DecodingMode.BestPath:
        for ind, symbols_probability in enumerate(symbols_probabilities):
            word_embedding = best_path_decoding(symbols_probability)

            word = convert_to_word(word_embedding)
            ground_truth = convert_to_word(labels[ind])

            accuracy_cer_value += char_error_rate(word,
                                                  ground_truth).item() * 100
            accuracy_value += torch.equal(word_embedding, labels[ind])
    elif mode == DecodingMode.BeamSearch:
        for ind, symbols_probability in enumerate(symbols_probabilities):
            word_embedding = beam_search_decoding(
                symbols_probability, beam_width)

            word = convert_to_word(word_embedding)
            ground_truth = convert_to_word(labels[ind])

            accuracy_cer_value += char_error_rate(word,
                                                  ground_truth).item() * 100
            accuracy_value += torch.equal(word_embedding, labels[ind])
    elif mode == DecodingMode.BeamSearchLM:
        for ind, symbols_probability in enumerate(symbols_probabilities):
            word_embedding = beam_search_decoding_with_LM(
                symbols_probability, LM, beam_width)

            word = convert_to_word(word_embedding)
            ground_truth = convert_to_word(labels[ind])

            accuracy_cer_value += char_error_rate(word,
                                                  ground_truth).item() * 100
            accuracy_value += torch.equal(word_embedding, labels[ind])
    accuracy.value = accuracy_value
    accuracy_cer.value = accuracy_cer_value / symbols_probabilities.shape[0]


def train_model(epochs):
    device, data_loader_args = make_initial_setup()
    train_loader, test_loader = create_data_loaders(config.DATA_PATH, config.LABELS_FILE,
                                                  config.TESTING_PERCENT, config.BATCH_SIZE,
                                                  data_loader_args)
    parameters_file = join(config.DATA_PATH, config.SAVED_PARAMETERS)
    log_file = join(config.DATA_PATH, config.LOG_FILE)

    net = SimpleHTR(parameters_file, device).to(device)
    LM = LanguageModel(config.DATA_PATH, config.TEXT_FILE)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    loss_function = torch.nn.CTCLoss(blank=0, reduction='mean')
    epoch_start = 1
    step_start = 0

    # if we have already stored the intermediate results
    if isfile(parameters_file):
        state = torch.load(parameters_file)
        net.load_previous_state(state)
        optimizer.load_state_dict(state['optimizer'])
        epoch_start = state['epoch'] + 1
        step_start = state['step']

    with open(log_file, 'a') as log_file_out:
        if not step_start:
            print('Mode', 'Step', 'Accuracy Best Path', 'Accuracy Best Path CER', 'Accuracy Beam Search',
                  'Accuracy Beam Search CER', 'Accuracy Beam Search with LM', 'Accuracy Beam Search with LM CER', 'Loss', file=log_file_out, sep=',')
        step = step_start
        for epoch in range(epoch_start, epochs + 1):
            for images, labels, lengths in train_loader:
                images, labels, lengths = images.to(device), labels.to(device), lengths.to(device)

                optimizer.zero_grad()
                images_transformed = net.forward(images)

                input_lengths = torch.full(
                    size=[images_transformed.shape[0]], 
                    fill_value=config.MAX_LABEL_LENGTH, 
                    dtype=torch.int32, device=device)

                symbols_probabilities = torch.permute(
                    images_transformed, (2, 0, 1))

                # required for CTCLoss and CTCDecoding
                symbols_probabilities = torch.nn.functional.log_softmax(
                    input=symbols_probabilities, dim=2)

                loss = loss_function(symbols_probabilities, labels,
                                     input_lengths, lengths)

                if (step % 500 == 0):
                    symbols_probabilities = symbols_probabilities.permute(
                        1, 2, 0)
                    
                    processes = list()
                    accuracy = [Value('f', 0.0) for _ in range(len(DecodingMode))]
                    accuracy_cer = [Value('f', 0.0) for _ in range(len(DecodingMode))]

                    symbols_probabilities = symbols_probabilities.detach().cpu()
                    labels = labels.detach().cpu()

                    for mode in DecodingMode:
                        process = Process(target=calculate_accuracy, args=(symbols_probabilities, labels, accuracy[mode.value],
                                                           accuracy_cer[mode.value], mode, 10, LM,), daemon=True)
                        processes.append(process)
                        process.start()

                    for process in processes:
                        process.join()

                    loss_val = loss.item()
                    print('Training', step, accuracy[DecodingMode.BestPath.value].value, accuracy_cer[DecodingMode.BestPath.value].value,
                          accuracy[DecodingMode.BeamSearch.value].value, accuracy_cer[DecodingMode.BeamSearch.value].value, 
                          accuracy[DecodingMode.BeamSearchLM.value].value, accuracy_cer[DecodingMode.BeamSearchLM.value].value,
                          loss_val, file=log_file_out, sep=',')
                    print(f"Training, Step {step}, accuracy_best_path {accuracy[DecodingMode.BestPath.value].value},", end=' ')
                    print(f"accuracy_best_path_cer {accuracy_cer[DecodingMode.BestPath.value].value},", end=' ')
                    print(f"accuracy_beam_search {accuracy[DecodingMode.BeamSearch.value].value},", end=' ')
                    print(f"accuracy_beam_search_cer {accuracy_cer[DecodingMode.BeamSearch.value].value},", end=' ')
                    print(f"accuracy_beam_search_with_LM {accuracy[DecodingMode.BeamSearchLM.value].value},", end=' ')
                    print(f"accuracy_beam_search_with_LM_cer {accuracy_cer[DecodingMode.BeamSearchLM.value].value},", end=' ')
                    print(f"loss {loss_val}.")

                    net.save(step, epoch, optimizer.state_dict())
                step += 1

                loss.backward()
                optimizer.step()

            temp_losses = list()
            temp_accuracy_best_path = list()
            temp_accuracy_best_path_cer = list()
            temp_accuracy_beam_search = list()
            temp_accuracy_beam_search_cer = list()
            temp_accuracy_beam_search_with_LM = list()
            temp_accuracy_beam_search_with_LM_cer = list()

            net.eval()
            with torch.no_grad():
                for images, labels, lengths in test_loader:
                    images, labels, lengths = images.to(
                        device), labels.to(device), lengths.to(device)
                    images_transformed = net.forward(images)

                    input_lengths = torch.full(
                        size=[images_transformed.shape[0]], 
                        fill_value=config.MAX_LABEL_LENGTH, 
                        dtype=torch.int32, 
                        device=device)

                    symbols_probabilities = torch.permute(
                        images_transformed, (2, 0, 1))
                    symbols_probabilities = torch.nn.functional.log_softmax(
                        input=symbols_probabilities, dim=2)

                    loss = loss_function(symbols_probabilities, labels,
                                         input_lengths, lengths)

                    symbols_probabilities = symbols_probabilities.permute(
                        1, 2, 0)
                    
                    processes = list()
                    accuracy = [Value('f', 0.0)
                                for _ in range(len(DecodingMode))]
                    accuracy_cer = [Value('f', 0.0)
                                    for _ in range(len(DecodingMode))]

                    symbols_probabilities = symbols_probabilities.detach().cpu()
                    labels = labels.detach().cpu()

                    for mode in DecodingMode:
                        process = Process(target=calculate_accuracy, args=(symbols_probabilities, labels, accuracy[mode.value],
                                                                           accuracy_cer[mode.value], mode, 35, LM,), daemon=True)
                        processes.append(process)
                        process.start()

                    for process in processes:
                        process.join()

                    temp_losses.append(loss.item())
                    temp_accuracy_best_path.append(
                        accuracy[DecodingMode.BestPath.value].value)
                    temp_accuracy_best_path_cer.append(
                        accuracy_cer[DecodingMode.BestPath.value].value)
                    temp_accuracy_beam_search.append(
                        accuracy[DecodingMode.BeamSearch.value].value)
                    temp_accuracy_beam_search_cer.append(
                        accuracy_cer[DecodingMode.BeamSearch.value].value)
                    temp_accuracy_beam_search_with_LM.append(
                        accuracy[DecodingMode.BeamSearchLM.value].value)
                    temp_accuracy_beam_search_with_LM_cer.append(
                        accuracy_cer[DecodingMode.BeamSearchLM.value].value)
            net.train()

            mean_loss_value = np.mean(temp_losses)
            mean_accuracy_best_path_value = np.mean(temp_accuracy_best_path)
            mean_accuracy_best_path_value_cer = np.mean(temp_accuracy_best_path_cer)
            mean_accuracy_beam_search_value = np.mean(
                temp_accuracy_beam_search)
            mean_accuracy_beam_search_value_cer = np.mean(
                temp_accuracy_beam_search_cer)
            mean_accuracy_beam_search_with_LM_value = np.mean(
                temp_accuracy_beam_search_with_LM)
            mean_accuracy_beam_search_with_LM_value_cer = np.mean(
                temp_accuracy_beam_search_with_LM_cer)

            print('Testing', epoch, mean_accuracy_best_path_value, mean_accuracy_best_path_value_cer,
                  mean_accuracy_beam_search_value, mean_accuracy_beam_search_value_cer,
                  mean_accuracy_beam_search_with_LM_value, mean_accuracy_beam_search_with_LM_value_cer,
                  mean_loss_value, file=log_file_out, sep=',')
            print(f"Testing, Epoch {epoch}, accuracy_best_path {mean_accuracy_best_path_value},", end=' ')
            print(f"accuracy_best_path_cer {mean_accuracy_best_path_value_cer},", end=' ')
            print(f"accuracy_beam_search {mean_accuracy_beam_search_value},", end=' ')
            print(f"accuracy_beam_search_cer {mean_accuracy_beam_search_value_cer},", end=' ')
            print(f"accuracy_beam_search_with_LM {mean_accuracy_beam_search_with_LM_value},", end=' ')
            print(f"accuracy_beam_search_with_LM_cer {mean_accuracy_beam_search_with_LM_value_cer},", end=' ')
            print(f"loss {mean_loss_value}.")

            net.save(step, epoch, optimizer.state_dict())


def predict(
        symbols_probabilities:torch.Tensor,
        mode:DecodingMode, 
        LM:LanguageModel=None,
        beam_width:int=25,
        lm_influence:float=config.LM_INFLUENCE):
    if mode == DecodingMode.BestPath:
        for symbols_probability in symbols_probabilities:
            word_embedding = best_path_decoding(symbols_probability)
    elif mode == DecodingMode.BeamSearch:
        for symbols_probability in symbols_probabilities:
            word_embedding = beam_search_decoding(
                symbols_probability, beam_width)
    elif mode == DecodingMode.BeamSearchLM:
        for symbols_probability in symbols_probabilities:
            word_embedding = beam_search_decoding_with_LM(
                symbols_probability, LM, beam_width, lm_influence)
    return word_embedding


def test_model(file_path:str, new:bool, beam_width:int=25, lm_influence:float=config.LM_INFLUENCE):
    device,_ = make_initial_setup()
    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE) 

    if new:
        image = cv.dilate(image, kernel=np.ones((3, 3), np.uint8), iterations=3)
        image = cv.erode(image, kernel=np.ones((5, 5), np.uint8), iterations=2)
        # image = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    image = torch.FloatTensor(image)
    image.unsqueeze_(dim=0)
    image = preprocess_image(image, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    image -= torch.mean(image)
    image_std = torch.std(image)
    image = image / image_std if image_std > 0 else image
    image.unsqueeze_(dim=0)
    image = image.to(device)

    LM = LanguageModel(config.DATA_PATH, config.TEXT_FILE)

    parameters_file = join(config.DATA_PATH, config.SAVED_PARAMETERS)
    net = SimpleHTR(parameters_file, device).to(device)

    net.load_previous_state(torch.load(parameters_file))
    net.eval()
    with torch.no_grad():
        image_preds = net.forward(image)
        symbols_probabilities = torch.nn.functional.log_softmax(
            input=image_preds, dim=1)

        for mode in DecodingMode:
            word_embedding = predict(
                symbols_probabilities, mode, LM, beam_width, lm_influence)
            word = ''
            for t in word_embedding:
                word += config.INDEXES_TO_TERMINALS[t]
            print(f"Recognized word is \"{word}\" by {mode}")


def create_parser():
    parser = argparse.ArgumentParser(description='Simple Text Recognition System now on Pytorch!',
                                     epilog='(c) Mamedov Timur 2023. Moscow State University.')
    subparsers = parser.add_subparsers(dest='mode')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('epochs', 
                              type=int,
                              help='The required number of epochs in training')

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('file_path',
                             help='The relative path to the required image')
    test_parser.add_argument('-n', 
                             '--new', 
                             action='store_true', 
                             default=False,
                             help='Should be used when you are passing your own image')
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

    return parser


if __name__=='__main__':
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    if namespace.mode == 'train':
        train_model(namespace.epochs)
    elif namespace.mode == 'test':
        test_model(namespace.file_path, 
                   new=namespace.new, 
                   beam_width=namespace.beam_width, 
                   lm_influence=namespace.lm_influence)
    else:
        print("Check a list of possible arguments.")
