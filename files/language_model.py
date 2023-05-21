import pandas as pd
from os.path import join, isfile

import config

class LanguageModel:
    def __init__(self, data_path, text_file):
        self.csv_file = join(data_path, config.LM_TABLE)
        if not isfile(self.csv_file):
            first_letter = dict()
            letter_ratio = dict()

            path_read = join(data_path, text_file)

            with open(path_read, 'r') as text_file:
                prev = ' '
                for line in text_file:
                    for c in line:
                        if prev == ' ':
                            first_letter[c] = first_letter.get(c, 0) + 1
                        else:
                            if prev not in letter_ratio:
                                letter_ratio[prev] = dict()
                            letter_ratio[prev][c] = letter_ratio[prev].get(
                                c, 0) + 1
                        prev = c

            first_number = sum(first_letter.values())
            for letter in first_letter.keys():
                first_letter[letter] /= first_number

            for prev in letter_ratio.keys():
                rel_prev_number = sum(letter_ratio[prev].values())
                for next in letter_ratio[prev]:
                    letter_ratio[prev][next] /= rel_prev_number

            self.table = pd.DataFrame(
                index=config.INDEXES_TO_TERMINALS[1:], columns=config.INDEXES_TO_TERMINALS[1:])
            for prev in letter_ratio:
                self.table.loc[prev, :].update(letter_ratio[prev])

            self.table['FIRST'] = [0. for _ in range(len(self.table.index))]
            self.table['FIRST'].update(first_letter)
            self.table.fillna(0., inplace=True)
                
            self.table.to_csv(self.csv_file)
        else:
            self.table = pd.read_csv(self.csv_file, index_col=[0])
    
    def show(self):
        print(self.table)
    
    def get_single_probability(self, letter_ind):
        return self.table.iloc[letter_ind - 1]['FIRST']
    
    def get_relative_probability(self, prev_ind, next_ind):
        return self.table.iloc[prev_ind - 1][next_ind - 1]