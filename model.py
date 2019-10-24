#!/usr/bin/env python
# coding: utf-8

# In[32]:


from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras.utils
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder

import numpy as np 
import pandas as pd 
import random
import sys
import io


# In[4]:


song = pd.read_csv('entre_dos_aguas.csv')
# display(song)


# In[5]:


notes = list(song['Note'])
notes = notes*5
durations = list(song['Duration'])
durations = durations*5


# In[6]:


mx_notes = max(notes)+1
mx_dur = max(durations)+1
how_much = 50


# In[7]:


seq_len = 20

X_notes = []
Y_notes = [] 

for i in range(0, len(notes)-seq_len, 1):
    seq_in = notes[i:i+seq_len]
    seq_out = notes[i + seq_len]
    
    X_notes.append(seq_in)
    Y_notes.append(seq_out)


# In[8]:


X_dur = []
Y_dur = []

for i in range(0, len(durations)-seq_len-1):
    seq_in = durations[i:i+seq_len]
    seq_out = durations[i + seq_len]
    
    X_dur.append(seq_in)
    Y_dur.append(seq_out)


# In[9]:


train_note = keras.utils.np_utils.to_categorical(X_notes)
test_note = keras.utils.np_utils.to_categorical(Y_notes)


# In[10]:


train_dur = keras.utils.np_utils.to_categorical(X_dur)
test_dur =  keras.utils.np_utils.to_categorical(Y_dur)


# In[11]:


print(train_note.shape)
print(test_note.shape)


# In[12]:


print(train_dur.shape)
print(test_dur.shape)


# In[13]:


model = Sequential()
model.add(LSTM(128, input_shape=(seq_len, mx_notes)))
model.add(Dense(mx_notes, activation='softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[14]:


dur_model = Sequential()
dur_model.add(LSTM(128, input_shape=(seq_len, mx_dur)))
dur_model.add(Dense(mx_dur, activation='softmax'))
dur_optimizer = RMSprop(learning_rate=0.01)
dur_model.compile(loss='categorical_crossentropy', optimizer=dur_optimizer)


# In[46]:


note_gen = []

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    start_index = random.randint(0, len(notes) - seq_len - 1)
    for diversity in [0.4 for i in range(5)]:
        print('----- diversity:', diversity)

        generated = []
        sentence = notes[start_index: start_index + seq_len]
        generated += sentence
        print('----- Generating with seed: ' + str(sentence))

        for i in range(how_much):
            x_pred = np.zeros((1, seq_len, mx_notes))
            for t, nt in enumerate(sentence):
                x_pred[0, t, nt] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)

            sentence = sentence[1:]
            sentence.append(next_index)
            note_gen.append(next_index)
    print(note_gen)


# In[47]:


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(train_note, test_note,
          batch_size=128,
          epochs=1,
          callbacks=[print_callback])


# In[48]:


dur_gen = []


def dur_on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    start_index = random.randint(0, len(durations) - seq_len - 1)
    for diversity in [0.6 for i in range(5)]:
        print('----- diversity:', diversity)

        generated = []
        sentence = durations[start_index: start_index + seq_len]
        generated += sentence
        print('----- Generating with seed: ' + str(sentence))

        for i in range(how_much):
            x_pred = np.zeros((1, seq_len, mx_dur))
            for t, nt in enumerate(sentence):
                x_pred[0, t, nt] = 1.

            preds = dur_model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)

            sentence = sentence[1:]
            sentence.append(next_index)
            dur_gen.append(next_index)
    print(dur_gen)


# In[49]:


dur_print_callback = LambdaCallback(on_epoch_end=dur_on_epoch_end)

dur_model.fit(train_dur, test_dur,
          batch_size=128,
          epochs=1,
          callbacks=[dur_print_callback])


# In[52]:


off_gen = [] 
accum = 0 

for i in range(len(dur_gen)): 
    off_gen.append(accum)
    accum += dur_gen[i]
    
music_gen = pd.DataFrame(list(zip(note_gen, dur_gen, off_gen)), columns =['Note', 'Duration', 'Offset'])


music_gen.to_csv('generated_music.csv', sep=',', index=False)

