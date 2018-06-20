'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.epochs = 10
config.batch_size = 33
config.dropout = 0.29
config.max_features = 2500




# max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
# batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=config.max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
labels = ["Positive", "Negative"]

print('Build model...')
model = Sequential()
model.add(Embedding(config.max_features, 128))
model.add(LSTM(128, dropout=config.dropout, recurrent_dropout=config.dropout))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# create a lambda callback to call log

print('Train...')
model.fit(x_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(x_test, y_test), 
          callbacks=[WandbCallback()])

score, acc = model.evaluate(x_test, y_test,
                            batch_size=config.batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
