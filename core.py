
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
from time import time

import checkpoints

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def load_dataset(ruta, categoria, modulo, dificultad, train_samples, interpolate_samples, extrapolate_samples, categoria_de_extrapolacion, modulo_de_extrapolacion, batch_size, num_workers):

    mdsmgr = MathDatasetManager(ruta)
    if dificultad == 'train-mixed':
        ds_easy = mdsmgr.build_dataset_from_module(categoria, modulo, 'train-easy', max_elements=train_samples)
        ds_medium = mdsmgr.build_dataset_from_module(categoria, modulo, 'train-medium', max_elements=train_samples)
        ds_hard = mdsmgr.build_dataset_from_module(categoria, modulo, 'train-hard', max_elements=train_samples) 
        ds = torch.utils.data.ConcatDataset([ds_easy, ds_medium, ds_hard])
        print("dataset loaded with size", len(ds))
    else:
        ds = mdsmgr.build_dataset_from_module(categoria, modulo, dificultad, max_elements=train_samples)
        print("dataset loaded with size", len(ds))
    
    ds_interpolate = mdsmgr.build_dataset_from_module( categoria, modulo, 'interpolate', max_elements=interpolate_samples )
    print("interpolate dataset size", len(ds_interpolate))
    
    ds_extrapolate = mdsmgr.build_dataset_from_module( categoria_de_extrapolacion, modulo_de_extrapolacion, 'extrapolate',  max_elements=extrapolate_samples)
    print("extrapolate dataset size", len(ds_extrapolate))
    # here we split data in 90/10% for train/validation and use interpolate for test
    train_ds, val_ds = math_dataset.random_split_dataset(ds, split_rate=0.9)

    # we provide the function question_answer_to_position_batch_collate_fn that collates
    # all questions/answers into transformer format enhanced with char positioning
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn)

    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn)

    interpolate_loader = data.DataLoader(
        ds_interpolate, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn)

    extrapolate_loader = data.DataLoader(
        ds_extrapolate, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=question_answer_to_position_batch_collate_fn)
    
    
    return train_loader, val_loader, interpolate_loader, extrapolate_loader



def build_model(device, tipo, n_layers):
    model = utils.build_Model(device=device, tipo=tipo, n_layers=n_layers)
    return model


def experiment(device, model, lr, exp_name, unique_id, epochs, train_loader, val_loader, interpolate_loader, extrapolate_loader):
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.995), eps=1e-9)
    tb = Tensorboard(exp_name, unique_name=unique_id)
    tiempo_inicial = time()
    model_process.train(
        exp_name, unique_id,
        model, 
        train_loader, val_loader, interpolate_loader, extrapolate_loader,
        optimizer, device,
        epochs=epochs, tb=tb, log_interval=100,
        )
    tiempo_final = time()
    print("Tiempo/época", (tiempo_final-tiempo_inicial)/epochs, 'ms.')
    print("Tiempo/batch", ((tiempo_final - tiempo_inicial))/(epochs*(len(train_loader)+ len(val_loader) + len(interpolate_loader)+len(extrapolate_loader))), 'ms.')
    
    loss_valid = go.Figure()
    acc_valid = go.Figure()
    loss_inter = go.Figure()
    acc_inter = go.Figure()
    
    valid = tensorboard_event_accumulator(
        'runs/' + f"{exp_name}_{unique_id}" + "_eval"
    )

    interpolate = tensorboard_event_accumulator(
        'runs/' + f"{exp_name}_{unique_id}" + "_interpolate"
    )
    
    valid_accuracy = valid.Scalars("epoch/accuracy")
    valid_loss_per_char = valid.Scalars("epoch/loss_per_char")
    
    interpolate_accuracy = interpolate.Scalars("epoch/accuracy")
    interpolate_loss_per_char = interpolate.Scalars("epoch/loss_per_char")
    
    x_valid = list(map(lambda l: l.step, valid_loss_per_char))
    y_valid = list(map(lambda l: l.value, valid_loss_per_char))
    y_accuracy_valid = list(map(lambda l: l.value, valid_accuracy))
    x_interpolate = list(map(lambda l: l.step, interpolate_loss_per_char))
    y_interpolate = list(map(lambda l: l.value, interpolate_loss_per_char))
    y_accuracy_interpolate = list(map(lambda l: l.value, interpolate_accuracy))
    
    loss_valid.add_trace(go.Scatter(x=x_valid[::2], y=y_valid[::2],
                         mode='lines'))
    acc_valid.add_trace(go.Scatter(x=x_valid[::2], y=y_accuracy_valid[::2],
                         mode='lines'))
    loss_inter.add_trace(go.Scatter(x=x_interpolate, y=y_interpolate,
                         mode='lines'))
    acc_inter.add_trace(go.Scatter(x=x_interpolate, y=y_accuracy_interpolate,
                         mode='lines'))
    loss_valid.update_layout(title='Loss per epoch (Valid)', title_x=0.45,
                           xaxis_title='epoch',
                  yaxis_title='Loss')

    loss_valid.show()

    acc_valid.update_layout(title='Accuracy per epoch (Valid)', title_x=0.45,
                               xaxis_title='epoch',
                      yaxis_title='Accuracy')

    acc_valid.show()

    loss_inter.update_layout(title='Loss per epoch (inter)', title_x=0.45,
                               xaxis_title='epoch',
                      yaxis_title='Loss')

    loss_inter.show()


    loss_valid = go.Figure()

    acc_inter.update_layout(title='Accuracy per epoch (inter)', title_x=0.45,
                               xaxis_title='epoch',
                      yaxis_title='Accuracy')

    acc_inter.show()
    
    
def multiple(lista_experimentos, names):
    loss_valid = go.Figure()
    acc_valid = go.Figure()
    loss_inter = go.Figure()
    acc_inter = go.Figure()
    for experimento, name in zip(lista_experimentos, names):
       

        valid = tensorboard_event_accumulator(
            'runs/' + experimento + "_eval"
        )

        interpolate = tensorboard_event_accumulator(
            'runs/' + experimento + "_interpolate"
        )

        valid_accuracy = valid.Scalars("epoch/accuracy")
        valid_loss_per_char = valid.Scalars("epoch/loss_per_char")

        interpolate_accuracy = interpolate.Scalars("epoch/accuracy")
        interpolate_loss_per_char = interpolate.Scalars("epoch/loss_per_char")

        x_valid = list(map(lambda l: l.step, valid_loss_per_char))
        y_valid = list(map(lambda l: l.value, valid_loss_per_char))
        y_accuracy_valid = list(map(lambda l: l.value, valid_accuracy))
        x_interpolate = list(map(lambda l: l.step, interpolate_loss_per_char))
        y_interpolate = list(map(lambda l: l.value, interpolate_loss_per_char))
        y_accuracy_interpolate = list(map(lambda l: l.value, interpolate_accuracy))

        loss_valid.add_trace(go.Scatter(x=x_valid[::2], y=y_valid[::2],
                             mode='lines',name=name))
        acc_valid.add_trace(go.Scatter(x=x_valid[::2], y=y_accuracy_valid[::2],
                             mode='lines',name=name))
        loss_inter.add_trace(go.Scatter(x=x_interpolate, y=y_interpolate,
                             mode='lines',name=name))
        acc_inter.add_trace(go.Scatter(x=x_interpolate, y=y_accuracy_interpolate,
                             mode='lines',name=name))
        
        
    loss_valid.update_layout(title='Loss per epoch (Valid)', title_x=0.45,
                           xaxis_title='epoch',
                  yaxis_title='Loss')

    loss_valid.show()

    acc_valid.update_layout(title='Accuracy per epoch (Valid)', title_x=0.45,
                               xaxis_title='epoch',
                      yaxis_title='Accuracy')

    acc_valid.show()

    loss_inter.update_layout(title='Loss per epoch (inter)', title_x=0.45,
                               xaxis_title='epoch',
                      yaxis_title='Loss')

    loss_inter.show()


    loss_valid = go.Figure()

    acc_inter.update_layout(title='Accuracy per epoch (inter)', title_x=0.45,
                               xaxis_title='epoch',
                      yaxis_title='Accuracy')

    acc_inter.show()