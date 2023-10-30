### Do not change this parameters ###
MAX_LABEL_LENGTH = 100
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 800
#####################################

DATA_PATH = '..\data'
RAW_LABELS_FILE = 'words.txt'
LABELS_FILE = 'labels.txt'
TEST_FILE = 'test.txt'
VAL_FILE = 'val.txt'
TRAIN_FILE = 'train.txt'

PARAMS = '.\params'
TEXT_FILE = 'corpus.txt'
LM_TABLE = 'LMTable.csv'

TERMINALS_NUMBER = 79
TERMINALS_TO_INDEXES = {'A': 1,  'M': 2,  'O': 3,  'V': 4,  'E': 5,  't': 6,
            'o': 7,  's': 8,  'p': 9,  'r': 10, '.': 11, 'G': 12,
            'a': 13, 'i': 14, 'k': 15, 'e': 16, 'l': 17, 'f': 18,
            'm': 19, 'n': 20, 'g': 21, 'y': 22, 'L': 23, 'b': 24,
            'u': 25, 'P': 26, 'd': 27, 'w': 28, 'h': 29, 'j': 30,
            'c': 31, ',': 32, ' ': 33, 'x': 34, '0': 35, 'F': 36,
            'W': 37, 'T': 38, '-': 39, "'": 40, 'v': 41, 'B': 42,
            'H': 43, '"': 44, 'S': 45, '1': 46, '9': 47, '5': 48,
            '8': 49, '3': 50, '#': 51, 'q': 52, 'N': 53, 'R': 54,
            'D': 55, 'K': 56, 'U': 57, 'I': 58, 'C': 59, '(': 60,
            '4': 61, ')': 62, '2': 63, ':': 64, 'J': 65, 'Y': 66,
            '7': 67, ';': 68, 'z': 69, 'Z': 70, '6': 71, '?': 72,
            '*': 73, 'X': 74, 'Q': 75, '!': 76, '/': 77, '&': 78,
            '+': 79}

INDEXES_TO_TERMINALS = ['', 'A', 'M', 'O', 'V', 'E', 't', 'o', 's', 'p', 'r', '.', 'G',
            'a', 'i', 'k', 'e', 'l', 'f', 'm', 'n', 'g', 'y', 'L', 'b', 'u',
            'P', 'd', 'w', 'h', 'j', 'c', ',', ' ', 'x', '0', 'F', 'W', 'T',
            '-', "'", 'v', 'B', 'H', '"', 'S', '1', '9', '5', '8', '3', '#',
            'q', 'N', 'R', 'D', 'K', 'U', 'I', 'C', '(', '4', ')', '2', ':',
            'J', 'Y', '7', ';', 'z', 'Z', '6', '?', '*', 'X', 'Q', '!', '/',
            '&', '+']

BLANK = 0

TESTING_PERCENT = 0.1
VALIDATION_PERCENT = 0.2
BATCH_SIZE = 32

'''
BEST_PARAMS = 'Inv_Lin_Trans_Less_Best.pth'
CUR_PARAMS = 'Inv_Lin_Trans_Less.pth'

BEST_PARAMS = 'Inv_Lin_Best.pth'
CUR_PARAMS = 'Inv_Lin.pth'

BEST_PARAMS = 'T42_modes_Best.pth'
CUR_PARAMS = 'T42_modes.pth'
'''

BEST_PARAMS = 'CRNN_BEST.pth'
CUR_PARAMS = 'CRNN.pth'
LOG_FILE = 'CRNN.txt'

LEARNING_RATE = 3e-4
LM_INFLUENCE = 1e-1

## Baseline options ##
class BASELINE:
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 128

    MAX_LABEL_LENGTH = 32

    BEST_PARAMS = 'base_best.pth'
    PARAMS = 'base.pth'
    LOG_FILE = 'base.txt'

    LR = 1e-4