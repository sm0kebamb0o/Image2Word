import torch
from enum import Enum
from collections import defaultdict
import numpy as np
from math import inf

import threading

import Config


from LanguageModel import LanguageModel

class DecodingMode(Enum):
    BestPath = 1
    BeamSearch = 2
    BeamSearchLM = 3


def best_path_decoding(symbols_probability:torch.Tensor, device:torch.device):
    most_probable_symbols = torch.argmax(symbols_probability, dim=0)
    most_probable_label = torch.unique_consecutive(most_probable_symbols)
    most_probable_label = most_probable_label[most_probable_label != 0]

    new_label = torch.zeros(
        size=(Config.MaxLabelLength,), dtype=torch.long, device=device)
    new_label[:most_probable_label.shape[0]] = most_probable_label

    return new_label

def beam_search_decoding(
        symbols_probability:torch.Tensor, 
        beam_width:int=3, 
        device:torch.device=torch.device('cpu')):
    """
        Implements beam search decoding approach to the probabilities matrix
    """
    blank_idx = 0
    not_probable = -10.

    class BeamInfo:
        def __init__(self):
            self.total_probability = not_probable
            self.blank_ending_probability = not_probable
            self.non_blank_ending_probability = not_probable

    beams = defaultdict(BeamInfo)
    beams[tuple()] = BeamInfo()
    beams[tuple()].blank_ending_probability = 0.
    beams[tuple()].total_probability = 0.
    (symbols_number, time_stamps_number) = symbols_probability.shape

    for t in range(time_stamps_number):
        current_beams = defaultdict(BeamInfo)

        most_probable_beams = sorted(beams.keys(), reverse=True,
                                     key=lambda beam: beams[beam].total_probability)[:beam_width]

        for beam in most_probable_beams:
            if len(beam):
                # path ending on the same letter probability
                non_blank_ending_probability = beams[beam].non_blank_ending_probability + \
                    symbols_probability[beam[-1], t].item()
            else:
                non_blank_ending_probability = -inf

            # blank ending
            blank_ending_probability = beams[beam].total_probability + \
                symbols_probability[blank_idx, t].item()

            current_beams[beam].total_probability = np.logaddexp(
                current_beams[beam].total_probability, np.logaddexp(
                    non_blank_ending_probability, blank_ending_probability))

            current_beams[beam].blank_ending_probability = np.logaddexp(
                current_beams[beam].blank_ending_probability, blank_ending_probability)

            current_beams[beam].non_blank_ending_probability = np.logaddexp(
                current_beams[beam].non_blank_ending_probability, non_blank_ending_probability)

            for s in range(1, symbols_number):
                new_beam = beam + (s,)
                if len(beam) > 0 and beam[-1] == new_beam[-1]:
                    non_blank_ending_probability = beams[beam].blank_ending_probability + \
                        symbols_probability[s, t].item()
                else:
                    non_blank_ending_probability = beams[beam].total_probability + \
                        symbols_probability[s, t].item()

                current_beams[new_beam].total_probability = np.logaddexp(
                    current_beams[new_beam].total_probability,
                    non_blank_ending_probability)
                current_beams[new_beam].non_blank_ending_probability = np.logaddexp(
                    current_beams[new_beam].non_blank_ending_probability, non_blank_ending_probability)
        beams = current_beams
    word = sorted(beams.keys(), reverse=True,
                  key=lambda beam: beams[beam].total_probability)[0]
    word_embedding = torch.zeros(
        size=(Config.MaxLabelLength,), dtype=torch.long, device=device)
    word_embedding[:len(word)] = torch.LongTensor(word)
    return word_embedding


def beam_search_decoding_with_LM(
        symbols_probability:torch.Tensor, 
        LM:LanguageModel, 
        beam_width:int=3, 
        device:torch.device='cpu'):
    blank_idx = 0
    not_probable = -10.
    LM_influence = 0.1

    class BeamInfo:
        def __init__(self):
            self.total_probability = not_probable
            self.blank_ending_probability = not_probable
            self.non_blank_ending_probability = not_probable
            self.word_probability = 0.
            self.LM_applied = False

    beams = defaultdict(BeamInfo)
    beams[tuple()] = BeamInfo()
    beams[tuple()].blank_ending_probability = 0.
    beams[tuple()].total_probability = 0.
    (symbols_number, time_stamps_number) = symbols_probability.shape

    for t in range(time_stamps_number):
        current_beams = defaultdict(BeamInfo)

        most_probable_beams = sorted(beams.keys(), reverse=True,
                                     key=lambda beam: beams[beam].total_probability + \
                                        beams[beam].word_probability / max(len(beam), 1.))[:beam_width]
        
        for beam in most_probable_beams:
            if len(beam):
                # path ending on the same letter probability
                non_blank_ending_probability = beams[beam].non_blank_ending_probability + \
                    symbols_probability[beam[-1], t].item()
            else:
                non_blank_ending_probability = -inf

            # blank ending
            blank_ending_probability = beams[beam].total_probability + \
                symbols_probability[blank_idx, t].item()

            current_beams[beam].total_probability = np.logaddexp(
                current_beams[beam].total_probability, np.logaddexp(
                    non_blank_ending_probability, blank_ending_probability))

            current_beams[beam].blank_ending_probability = np.logaddexp(
                current_beams[beam].blank_ending_probability, blank_ending_probability)

            current_beams[beam].non_blank_ending_probability = np.logaddexp(
                current_beams[beam].non_blank_ending_probability, non_blank_ending_probability)
            
            current_beams[beam].word_probability = beams[beam].word_probability
            
            current_beams[beam].LM_applied = True

            for s in range(1, symbols_number):
                new_beam = beam + (s,)
                if len(beam) > 0 and beam[-1] == new_beam[-1]:
                    non_blank_ending_probability = beams[beam].blank_ending_probability + \
                        symbols_probability[s, t].item()
                else:
                    non_blank_ending_probability = beams[beam].total_probability + \
                        symbols_probability[s, t].item()

                current_beams[new_beam].total_probability = np.logaddexp(
                    current_beams[new_beam].total_probability,
                    non_blank_ending_probability)
                current_beams[new_beam].non_blank_ending_probability = np.logaddexp(
                    current_beams[new_beam].non_blank_ending_probability, non_blank_ending_probability)
                
                if current_beams[new_beam].LM_applied:
                    continue

                current_beams[new_beam].LM_applied = True

                if len(new_beam) == 1:
                    probability = LM.get_single_probability(new_beam[-1])
                else:
                    probability = LM.get_relative_probability(new_beam[-2], new_beam[-1])
                with np.errstate(divide='ignore'):
                    current_beams[new_beam].word_probability = beams[beam].word_probability + \
                        LM_influence * np.log(float(probability))

        beams = current_beams

    word = sorted(beams.keys(), reverse=True,
                  key=lambda beam: beams[beam].total_probability + beams[beam].word_probability / 
                  max(len(beam), 1.))[0]
    
    word_embedding = torch.zeros(
        size=(Config.MaxLabelLength,), dtype=torch.long, device=device)
    word_embedding[:len(word)] = torch.LongTensor(word)
    return word_embedding