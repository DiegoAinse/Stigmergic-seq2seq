#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import torch
from torch.utils import data
import torch.optim as optim

import math_dataset
from math_dataset import MathDatasetManager

from models.Models import Transformer
from math_dataset import (
    random_split_dataset,
    question_answer_to_position_batch_collate_fn
)
import model_process
import utils
from tensorboard_utils import Tensorboard
from tensorboard_utils import tensorboard_event_accumulator

import checkpoints

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'notebook')

print("Torch Version", torch.__version__)

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

mdsmgr = MathDatasetManager(
  "/media/sergio/traballos sergio/msi/datasets/mathematics_dataset-v1.0"
)

seed = 1
torch.manual_seed(seed)
device = torch.device("cpu")
print("device", device)

#tipos: 'stigmergic-enc-dec', 'stigmergic-output', 'stigmergic-environment', 'Transformer', 'stigmergic-total'
#categorias y módulos de extrapolación: 'algebra' ('polinomial_roots_big'), 'arithmetic' ('add_or_sub_big', 'add_or_sub_multiple_longer', 'div_big', 'mixed_longer', 'mul_big', 'mul_div_multiple_longer'), 'comparison' ('closest_more', 'kth_biggest_more', 'sort_more'), 'measurement' ('conversion'), 'numbers' ('place_value_big', 'round_number_big'), 'probability' ('swr_p_level_set_more_samples', 'swr_p_sequence_set_more_samples')
 
tipo = 'stigmergic-output'
n_layers = 2
categoria = 'algebra'
modulo = 'linear_1d'
dificultad = 'train-easy'
categoria_de_extrapolacion = 'arithmetic'
modulo_de_extrapolacion = 'add_or_sub_big'
unique_id = "1-2"

exp_name = tipo + '_'  + str(n_layers)+ 'layers_' + categoria + '_' + modulo + '_' + dificultad

ds = mdsmgr.build_dataset_from_module(categoria, modulo, dificultad, max_elements=5000)
print("size", len(ds))

ds_interpolate = mdsmgr.build_dataset_from_module(
    categoria, modulo, 'interpolate', max_elements=500
)
print("interpolate dataset size", len(ds_interpolate))

ds_extrapolate = mdsmgr.build_dataset_from_module(
    categoria_de_extrapolacion, modulo_de_extrapolacion, 'extrapolate',  max_elements=500
)
print("extrapolate dataset size", len(ds_extrapolate))

model = utils.build_Model(tipo = tipo, n_layers = n_layers)
model

optimizer = optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.995), eps=1e-9)

# here we split data in 90/10% for train/validation and use interpolate for test
train_ds, val_ds = math_dataset.random_split_dataset(ds, split_rate=0.9)

# we provide the function question_answer_to_position_batch_collate_fn that collates
# all questions/answers into transformer format enhanced with char positioning
train_loader = data.DataLoader(
    train_ds, batch_size=10, shuffle=True, num_workers=4,
    collate_fn=question_answer_to_position_batch_collate_fn)

val_loader = data.DataLoader(
    val_ds, batch_size=10, shuffle=False, num_workers=4,
    collate_fn=question_answer_to_position_batch_collate_fn)

interpolate_loader = data.DataLoader(
    ds_interpolate, batch_size=10, shuffle=False, num_workers=4,
    collate_fn=question_answer_to_position_batch_collate_fn)

extrapolate_loader = data.DataLoader(
    ds_extrapolate, batch_size=10, shuffle=False, num_workers=4,
    collate_fn=question_answer_to_position_batch_collate_fn)

tb = Tensorboard(exp_name, unique_name=unique_id)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
        
model = model.to(device)

model_process.train(
    exp_name, unique_id,
    model, 
    train_loader, val_loader, interpolate_loader, extrapolate_loader,
    optimizer, device,
    epochs=10, tb=tb, log_interval=100,
)

