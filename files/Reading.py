import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

import config

df = pd.read_csv(join(config.DATA_PATH, config.LOG_FILE))

df['Accuracy Best Path'] *= 2
df['Accuracy Beam Search'] *= 2
df['Accuracy Beam Search with LM'] *= 2

train = df[df['Mode'] == 'Training'].copy()
test = df[df['Mode'] == 'Testing'].copy()

test.drop(columns=['Mode'], inplace=True)
test.rename(columns={'Step':'Epoch'}, inplace=True)
test.set_index('Epoch', inplace=True)

test['Loss'].plot(kind='line',color='red', grid=True)
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.title('Loss value on Testing')
plt.show()

test[['Accuracy Best Path','Accuracy Beam Search','Accuracy Beam Search with LM']].plot(kind='line', grid=True)
plt.legend()
plt.title('Accuracy on Testing')
plt.ylabel('Accuracy')
plt.axis([1, 12, 0, 100])
plt.show()

test[['Accuracy Best Path CER', 'Accuracy Beam Search CER',
      'Accuracy Beam Search with LM CER']].plot(kind='line', grid=True)
plt.legend()
plt.title('CER on Testing')
plt.ylabel('CER')
plt.show()


train.drop(columns=['Mode'], inplace=True)
train.set_index('Step', inplace=True)

train['Loss'].plot(kind='line', color='red', grid=True)
plt.ylabel('Loss value')
plt.title('Loss value on Training')
plt.show()

train[['Accuracy Best Path', 'Accuracy Beam Search',
       'Accuracy Beam Search with LM']].plot(kind='line', grid=True)
plt.ylabel('Accuracy')
plt.title('Accuracy on Training')
plt.show()

train[['Accuracy Best Path CER', 'Accuracy Beam Search CER',
       'Accuracy Beam Search with LM CER']].plot(kind='line', grid=True)
plt.ylabel('Accuracy')
plt.title('CER on Training')
plt.show()
