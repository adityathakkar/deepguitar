#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
import fractions
import decimal
import re
import seaborn as sns
import pickle

from music21 import stream, tempo, converter, corpus, instrument, midi, note, chord, pitch, musicxml, meter, key

plt.rcParams["figure.figsize"] = (20,10)


# In[5]:


# Load data from encoder script

encoded_part = pd.read_csv("generated_music.csv")

with open('tbl_entre_dos_aguas.pickle', 'rb') as handle:
    tbl = pickle.load(handle)
    
with open('inv_tbl_entre_dos_aguas.pickle', 'rb') as handle:
    inv_tbl = pickle.load(handle)
    
with open('lcm.pickle', 'rb') as handle:
    lcm = pickle.load(handle)
    


# In[ ]:


display(midi_stream)


# In[8]:


midi_stream = stream.Stream()
guitar_part = stream.Voice()
midi_stream.append(instrument.Guitar())

for index, row in encoded_part.iterrows():
    note_name = inv_tbl[row['Note']]
    if (note_name == 'REST'):
        nt = note.Rest()
    else: 
        if (' ' in note_name):
            nt = chord.Chord(note_name)
        else:
            nt = note.Note(note_name)
    nt.duration.quarterLength = float(row['Duration'])/lcm
    nt.offset = float(row['Offset'])/lcm
    guitar_part.append(nt)

# switch params to being loaded from pickle, instead of being hardcoded   
midi_stream.append(tempo.MetronomeMark(number=192.0))
midi_stream.append(meter.TimeSignature('4/4'))
midi_stream.append(key.Key('G'))


midi_stream.append(guitar_part)


# In[11]:


midi_stream.show('text')


# In[10]:


# midi_stream.show('midi')
fp = midi_stream.write('midi', fp='full_gen.mid')

