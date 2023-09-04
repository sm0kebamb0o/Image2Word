import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

import config

df = pd.read_csv(join(config.DATA_PATH, config.LOG_FILE))

df['Accuracy Best Path'] *= 2
df['Accuracy Beam Search'] *= 2
df['Accuracy Beam Search with LM'] *= 2

df.rename(columns={'Accuracy Best Path': 'Best Path',
                     'Accuracy Beam Search': 'Beam Search',
                     'Accuracy Beam Search with LM': 'Beam Search with LM'}, inplace=True)

train = df[df['Mode'] == 'Training'].copy()
test = df[df['Mode'] == 'Testing'].copy()

test.drop(columns=['Mode'], inplace=True)
test.rename(columns={'Step': 'Epoch'}, inplace=True)
test.set_index('Epoch', inplace=True)
max_epoch = test.index.max()

test['Loss'].plot(kind='line',color='red', grid=True)
plt.xlabel('Эпоха')
plt.ylabel('Значение функции потерь')
plt.title('Значение функции потерь на тестовой выборке')
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
# plt.axis([0, max_epoch + 1, 0, test['Loss'].max()*1.1])
plt.show()

test[['Best Path','Beam Search','Beam Search with LM']].plot(kind='line', grid=True)
plt.legend()
plt.title('Точность распознавания на тестовой выборке')
plt.ylabel('Точность распознавания')
plt.xlabel('Эпоха')
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
# plt.axis([0, max_epoch + 1, 0, 100])
plt.show()

plt.plot(test['Accuracy Best Path CER'], 'C0', label='Best Path')
plt.plot(test['Accuracy Beam Search CER'], 'C1', label='Beam Search')
plt.plot(test['Accuracy Beam Search with LM CER'], 'C2', label='Beam Search LM')

plt.legend()
plt.title('CER на тестовой выборке')
plt.ylabel('CER')
plt.xlabel('Эпоха')
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
# plt.axis([0, max_epoch + 1, 0, 100])
plt.show()


train.drop(columns=['Mode'], inplace=True)
train.set_index('Step', inplace=True)
max_step = train.index.max()

train['Loss'].plot(kind='line', color='red', grid=True)
plt.ylabel('Значение функции потерь')
plt.title('Значение функции потерь на обучающей выборке')
plt.xlabel('Итерация')
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.show()

train[['Best Path', 'Beam Search',
       'Beam Search with LM']].plot(kind='line', grid=True)
plt.ylabel('Точность распознавания')
plt.title('Точность распознавания на обучающей выборке')
plt.xlabel('Итерация')
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.show()

plt.plot(train['Accuracy Best Path CER'], 'C0', label='Best Path')
plt.plot(train['Accuracy Beam Search CER'], 'C1', label='Beam Search')
plt.plot(train['Accuracy Beam Search with LM CER'],
         'C2', label='Beam Search LM')
plt.ylabel('CER')
plt.title('CER на обучающей выборке')
plt.xlabel('Итерация')
plt.minorticks_on()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.legend()
plt.show()
