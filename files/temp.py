import fastwer
from torchmetrics.functional import char_error_rate

"""

# Define reference text and output text
ref = 'name'
output = 'nim'

# Obtain Sentence-Level Character Error Rate (CER)
cer = fastwer.score_sent(output, ref, char_level=True)
print(cer)

cer_2 = char_error_rate([output], [ref]).item()
print(cer_2)
"""
from main import convert_to_word
from data_normalizer import preprocess_image
from model import SimpleHTR
import config
from language_model import LanguageModel
from decoding import *

from multiprocessing import Process, Value
from torchvision.io import read_image, ImageReadMode
import torch
from os.path import join


def calculate_accuracy(symbols_probabilities: torch.Tensor,
                       labels: torch.Tensor,
                       accuracy:Value,
                       accuracy_cer:Value,
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
            print("Lm", word)
    accuracy.value = accuracy_value
    accuracy_cer.value = accuracy_cer_value / symbols_probabilities.shape[0]

if __name__=='__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    image = read_image('.\city.png', ImageReadMode.GRAY).to(torch.float)
    image = preprocess_image(image, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    image -= torch.mean(image)
    image_std = torch.std(image)
    image = image / image_std if image_std > 0 else image
    image = image.to(device)
    image.unsqueeze_(dim=0)

    LM = LanguageModel(config.DATA_PATH, config.TEXT_FILE)
    parameters_file = join(config.DATA_PATH, config.SAVED_PARAMETERS)

    net = SimpleHTR(parameters_file, device).to(device)
    net.load_previous_state(torch.load(parameters_file))
    net.eval()
    with torch.no_grad():
        image_preds = net.forward(image)

    symbols_probabilities = torch.nn.functional.log_softmax(
        input=image_preds, dim=1)

    word = 'Manchester'

    label = torch.zeros(size=(config.MAX_LABEL_LENGTH,), dtype=torch.long, device=device)
    for ind, symbol in enumerate(word):
        label[ind] = config.TERMINALS_TO_INDEXES[symbol]
    label.unsqueeze_(dim=0)
    print(label)

    processes = list()
    accuracy = [Value('f', 0.0) for _ in range(len(DecodingMode))]
    accuracy_cer = [Value('f', 0.0) for _ in range(len(DecodingMode))]

    symbols_probabilities = symbols_probabilities.detach().cpu()
    labels = label.detach().cpu()

    for mode in DecodingMode:
        process = Process(target=calculate_accuracy, args=(symbols_probabilities, labels, accuracy[mode.value],
                                                     accuracy_cer[mode.value], mode, 25, LM,), daemon=True)
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    print(accuracy[DecodingMode.BestPath.value].value)
    print(accuracy[DecodingMode.BeamSearch.value].value)
    print(accuracy[DecodingMode.BeamSearchLM.value].value)
