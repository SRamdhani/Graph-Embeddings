from graphs.main.model import Model
import tensorflow as tf
import json
import numpy as np
import math

tf.config.set_visible_devices([], 'GPU')

def random_example_batch(size):
    global full_data_x
    np.random.seed(100)
    r = np.random.choice(range(len(full_data_x)), size, replace=False)
    temp_1_x = []
    temp_1_x = [full_data_x[i] for i in r]
    return temp_1_x

class gen:
    def __init__(self, total_batches, batch_size, full_data_x, full_data_y):
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.full_data_x = full_data_x
        self.full_data_y = full_data_y

    def generator(self, i):
        temp_1_x = []
        temp_1_y = []
        if i == (self.total_batches-1):
            temp_1_x = self.full_data_x[(len(self.full_data_x)-self.batch_size):len(self.full_data_x)]
            temp_1_y = self.full_data_y[(len(self.full_data_y)-self.batch_size):len(self.full_data_y)]
        else:
            temp_1_x = self.full_data_x[(i*self.batch_size):(i*self.batch_size+self.batch_size)]
            temp_1_y = self.full_data_y[(i*self.batch_size):(i*self.batch_size+self.batch_size)]
        return np.array(temp_1_x), np.array(temp_1_y)

data_dict_vec = json.load(open('../descriptor_concept_data_dict_vec_NEW.json'))

### Save Vocab with retreivable indices for later. ##
voc_dict = {x: i for i, x in enumerate(list(data_dict_vec.keys()))}
voc_dict_reverse = {i: x for i, x in enumerate(list(data_dict_vec.keys()))}

### Convert data dic to dic of numeric. ###
data_dict_vec_numeric = {}
for key in data_dict_vec:
    new_key = voc_dict[key]
    new_list = [voc_dict[x] for x in data_dict_vec[key]]
    try:
        data_dict_vec_numeric[new_key] = new_list
    except:
        data_dict_vec_numeric[new_key] = []
        data_dict_vec_numeric[new_key] = new_list
    new_key = None
    new_list = None

### Create pseduo Skip-Gram. ###
pseduo_skip_gram_pairs = []
for key in data_dict_vec_numeric:
    [pseduo_skip_gram_pairs.append([key, x]) for x in data_dict_vec_numeric[key]]

### Split pseduo Skip-Gram to sep lists. ###
full_data_x = []
full_data_y = []
for element in pseduo_skip_gram_pairs:
    full_data_x.append(element[0])
    full_data_y.append([element[1]])

### Create simple batcher. ###
batch_size = 1000
num_epochs = 20000
embedding_size = 300
voc_size = len(voc_dict.keys())
num_sampled = 30
num_data = len(pseduo_skip_gram_pairs)
total_batches = num_data//batch_size + 1 if num_data % batch_size > 0 else num_data//batch_size

valid_word     = ['eye', 'carbamoylphosphate','colon','cochlear','posterior','circulation','buttocks','fractures']
valid_examples = [voc_dict[x] for x in valid_word]
valid_inputs   = tf.convert_to_tensor(valid_examples)

g = gen(total_batches, batch_size, full_data_x, full_data_y)

m = Model(num_epochs=num_epochs, total_batches=total_batches, num_sampled=num_sampled,
          batch_size=batch_size, embedding_size=embedding_size,voc_size=voc_size,
          learning_rate=1e-3)

m.train(top_k=8, g=g, valid_word=valid_word,
        valid_inputs=valid_inputs, threshold=5,
        voc_dic_reverse=voc_dict_reverse)
