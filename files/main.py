# imports
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import char_error_rate
from os.path import isfile, join
import sys
from multiprocessing import Process

from Model import CustomDataLoader, SimpleHTR
from Decoding import DecodingMode, best_path_decoding, beam_search_decoding, beam_search_decoding_with_LM
import Config
from LanguageModel import LanguageModel
from DataNormalizer import preprocess_image

def MakeInitialSetup():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        data_loader_args = {'num_workers': 1, 'pin_memory': True}
    else:
        device = torch.device('cpu')
        data_loader_args = {}

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    return device, data_loader_args


def CreateDataLoaders(
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
        word += Config.CharsInd[val]
    return word

def calculate_accuracy(
        symbols_probabilities:torch.Tensor, 
        labels:torch.Tensor, 
        device:torch.device, 
        mode:DecodingMode, 
        beam_width:int=10,
        LM:LanguageModel=None):
    accuracy_cer = 0.
    accuracy = 0.
    if mode == DecodingMode.BestPath:
        for ind, symbols_probability in enumerate(symbols_probabilities):
            word_embedding = best_path_decoding(symbols_probability, device)

            word = convert_to_word(word_embedding)
            ground_truth = convert_to_word(labels[ind])

            accuracy_cer += char_error_rate(word, ground_truth).item() * 100
            accuracy += torch.equal(word_embedding, labels[ind])
    elif mode == DecodingMode.BeamSearch:
        for ind, symbols_probability in enumerate(symbols_probabilities):
            word_embedding = beam_search_decoding(
                symbols_probability, beam_width, device)
            
            word = convert_to_word(word_embedding)
            ground_truth = convert_to_word(labels[ind])

            accuracy_cer += char_error_rate(word, ground_truth).item() * 100
            accuracy += torch.equal(word_embedding, labels[ind])
    elif mode == DecodingMode.BeamSearchLM:
        for ind, symbols_probability in enumerate(symbols_probabilities):
            word_embedding = beam_search_decoding_with_LM(
                symbols_probability, LM, beam_width, device)
            
            word = convert_to_word(word_embedding)
            ground_truth = convert_to_word(labels[ind])

            accuracy_cer += char_error_rate(word, ground_truth).item() * 100
            accuracy += torch.equal(word_embedding, labels[ind])
    return accuracy, accuracy_cer / symbols_probabilities.shape[0]


def train_model(epochs):
    device, data_loader_args = MakeInitialSetup()
    train_loader, test_loader = CreateDataLoaders(Config.DataPath, Config.LabelsFile,
                                                  Config.TestingPercent, Config.BatchSize,
                                                  data_loader_args)
    parameters_file = join(Config.DataPath, Config.SavedParameters)
    log_file = join(Config.DataPath, Config.LogFile)

    # Experiments
    net = SimpleHTR(parameters_file, device).to(device)
    LM = LanguageModel(Config.DataPath, Config.TextFile)
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
    loss_function = torch.nn.CTCLoss(blank=0, reduction='mean')
    epoch_start = 1
    step_start = 0

    # if we have already stored the intermediate results
    if isfile(parameters_file):
        state = torch.load(parameters_file)
        net.load_previous_state(state)
        optimizer.load_state_dict(state['optimizer'])
        epoch_start = state['epoch']
        step_start = state['step']

    # will be used in CTCLoss

    with open(log_file, 'a') as log_file_out:
        if not step_start:
            print('Mode', 'Step', 'Accuracy Best Path', 'Accuracy Best Path CER', 'Accuracy Beam Search',
                  'Accuracy Beam Search CER', 'Accuracy Beam Search with LM', 'Accuracy Beam Search with LM CER', 'Loss', file=log_file_out, sep=',')
        step = step_start
        for epoch in range(epoch_start, epochs + 1):
            for images, labels, lengths in train_loader:
                images, labels, lengths = images.to(
                    device), labels.to(device), lengths.to(device)

                optimizer.zero_grad()
                images_transformed = net.forward(images)

                input_lengths = torch.full(
                    size=[images_transformed.shape[0]], 
                    fill_value=Config.MaxLabelLength, 
                    dtype=torch.long, device=device)

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
                    for mode in DecodingMode:
                        process = Process(target=calculate_accuracy, args=(
                            symbols_probabilities, labels, device, mode, 10, LM), daemon=True)
                        processes.append(process)
                    accuracy_best_path, accuracy_best_path_cer = calculate_accuracy(symbols_probabilities,
                                                            labels, 
                                                            device, 
                                                            DecodingMode.BestPath)

                    accuracy_beam_search, accuracy_beam_search_cer = calculate_accuracy(symbols_probabilities,
                                                              labels, 
                                                              device, 
                                                              DecodingMode.BeamSearch,
                                                              10)
                    accuracy_beam_search_with_LM, accuracy_beam_search_with_LM_cer = calculate_accuracy(symbols_probabilities,
                                                                      labels, 
                                                                      device, 
                                                                      DecodingMode.BeamSearchLM, 
                                                                      10,
                                                                      LM)
                    loss_val = loss.item()
                    print('Training', step, accuracy_best_path, accuracy_best_path_cer,
                          accuracy_beam_search, accuracy_beam_search_cer, accuracy_beam_search_with_LM, accuracy_beam_search_with_LM_cer,
                          loss_val, file=log_file_out, sep=',')
                    print(
                        f"Training, Step {step}, accuracy_best_path {accuracy_best_path}, accuracy_best_path_cer {accuracy_best_path_cer}, \
                        accuracy_beam_search {accuracy_beam_search}, accuracy_beam_search_cer {accuracy_beam_search_cer}, \
                        accuracy_beam_search_with_LM {accuracy_beam_search_with_LM}, accuracy_beam_search_with_LM_cer {accuracy_beam_search_with_LM_cer}, loss {loss_val}.")

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
                        fill_value=Config.MaxLabelLength, 
                        dtype=torch.long, 
                        device=device)

                    symbols_probabilities = torch.permute(
                        images_transformed, (2, 0, 1))
                    symbols_probabilities = torch.nn.functional.log_softmax(
                        input=symbols_probabilities, dim=2)

                    loss = loss_function(symbols_probabilities, labels,
                                         input_lengths, lengths)

                    symbols_probabilities = symbols_probabilities.permute(
                        1, 2, 0)
                    accuracy_best_path, accuracy_best_path_cer = calculate_accuracy(symbols_probabilities,
                                                                                    labels,
                                                                                    device,
                                                                                    DecodingMode.BestPath)

                    accuracy_beam_search, accuracy_beam_search_cer = calculate_accuracy(symbols_probabilities,
                                                                                        labels,
                                                                                        device,
                                                                                        DecodingMode.BeamSearch, 20)
                    accuracy_beam_search_with_LM, accuracy_beam_search_with_LM_cer = calculate_accuracy(symbols_probabilities,
                                                                                                        labels,
                                                                                                        device,
                                                                                                        DecodingMode.BeamSearchLM,
                                                                                                        20,
                                                                                                        LM)
                    temp_losses.append(loss.item())
                    temp_accuracy_best_path.append(accuracy_best_path)
                    temp_accuracy_best_path_cer.append(accuracy_best_path_cer)
                    temp_accuracy_beam_search.append(accuracy_beam_search)
                    temp_accuracy_beam_search_cer.append(accuracy_beam_search_cer)
                    temp_accuracy_beam_search_with_LM.append(
                        accuracy_beam_search_with_LM)
                    temp_accuracy_beam_search_with_LM_cer.append(
                        accuracy_beam_search_with_LM_cer)
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
            print(
                f"Testing, Epoch {epoch}, accuracy_best_path {mean_accuracy_best_path_value}, accuracy_best_path_cer {mean_accuracy_best_path_value_cer}, \
                        accuracy_beam_search {mean_accuracy_beam_search_value}, accuracy_beam_search_cer {mean_accuracy_beam_search_value_cer}, \
                        accuracy_beam_search_with_LM {mean_accuracy_beam_search_with_LM_value}, accuracy_beam_search_with_LM_cer {mean_accuracy_beam_search_with_LM_value_cer}, \
                        loss {mean_loss_value}.")

            net.save(step, epoch, optimizer.state_dict())


def predict(
        symbols_probabilities:torch.Tensor,
        device:torch.Tensor, 
        mode:DecodingMode, 
        LM:LanguageModel=None):
    if mode == DecodingMode.BestPath:
        for symbols_probability in symbols_probabilities:
            word_embedding = best_path_decoding(symbols_probability, device)
    elif mode == DecodingMode.BeamSearch:
        for symbols_probability in symbols_probabilities:
            word_embedding = beam_search_decoding(
                symbols_probability, 20, device)
    elif mode == DecodingMode.BeamSearchLM:
        for symbols_probability in symbols_probabilities:
            word_embedding = beam_search_decoding_with_LM(
                symbols_probability, LM, 20, device)
    return word_embedding


def test_model(file_path:str):
    device,_ = MakeInitialSetup()
    image = torchvision.io.read_image(file_path,
                                      torchvision.io.ImageReadMode.GRAY).to(torch.float)
    image = preprocess_image(image, Config.ImageHeight, Config.ImageWidth)
    image -= torch.mean(image)
    image_std = torch.std(image)
    image = image / image_std if image_std > 0 else image
    image = image.to(device)
    image.unsqueeze_(dim=0)

    LM = LanguageModel(Config.DataPath, Config.TextFile)

    parameters_file = join(Config.DataPath, Config.SavedParameters)
    net = SimpleHTR(parameters_file, device).to(device)

    state = torch.load(parameters_file)
    net.load_state_dict(state['state_dict'])
    net.set_filter(state['filter'])
    net.eval()
    with torch.no_grad():
        image_preds = net.forward(image)
        symbols_probabilities = torch.permute(
            image_preds, (2, 0, 1))

        symbols_probabilities = torch.nn.functional.log_softmax(
            input=symbols_probabilities, dim=2)

        symbols_probabilities = symbols_probabilities.permute(1, 2, 0)

        for mode in DecodingMode:
            word_embedding = predict(
                symbols_probabilities, device, mode, LM)
            word = ''
            for t in word_embedding:
                word += Config.CharsInd[t]
            print(f"Recognized word is \"{word}\" by {mode}")


if __name__=='__main__':
    mode = sys.argv[1]
    if mode=='--train':
        train_model(int(sys.argv[2]))
    elif mode=='--test':
        file_path = sys.argv[2]
        test_model(file_path)