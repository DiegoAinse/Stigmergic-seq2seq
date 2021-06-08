#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

import torch

from models.torchsm import BaseLayer
from models.torchsm import StigmergicMemoryLayer

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
    
class StigmergicMemoryLayer_encoder(BaseLayer):
    def __init__(self, inputs, space_dim, **kwargs):
        BaseLayer.__init__(self, inputs, space_dim)

        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else 0
        self.hidden_dim = kwargs["hidden_dim"] if "hidden_layers" in kwargs else None

        self.n_inputs = inputs
        self.space_dim = self.n_outputs

        self.init_space = torch.zeros(self.space_dim)

        self.mark_net = torch.nn.Sequential()
        self.tick_net = torch.nn.Sequential()

        if self.hidden_layers != 0:
            self.mark_net.add_module("input_w", torch.nn.Linear(self.n_inputs, self.hidden_dim))
            self.tick_net.add_module("input_w", torch.nn.Linear(self.n_inputs, self.hidden_dim))
            
            self.mark_net.add_module("input_s", torch.nn.PReLU())
            self.tick_net.add_module("input_s", torch.nn.PReLU())
            
            for i in range(0, self.hidden_layers-1):
                self.mark_net.add_module("l"+str(i)+"_w", torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                self.tick_net.add_module("l"+str(i)+"_w", torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                
                self.mark_net.add_module("l"+str(i)+"_s", torch.nn.PReLU())
                self.tick_net.add_module("l"+str(i)+"_s", torch.nn.PReLU())
            
            self.mark_net.add_module("output_w", torch.nn.Linear(self.hidden_dim, self.space_dim))
            self.tick_net.add_module("output_w", torch.nn.Linear(self.hidden_dim, self.space_dim))

            self.mark_net.add_module("output_relu", torch.nn.PReLU())
            self.tick_net.add_module("output_relu", torch.nn.PReLU())
        else:
            self.mark_net.add_module("linear", torch.nn.Linear(self.n_inputs, self.space_dim))
            self.tick_net.add_module("linear", torch.nn.Linear(self.n_inputs, self.space_dim))

            self.mark_net.add_module("output_relu", torch.nn.PReLU())
            self.tick_net.add_module("output_relu", torch.nn.PReLU())
            
        self.reset()


    def forward(self, input_mark, input_tick = None):

        mark = self.mark_net(input_mark)
        tick = self.tick_net(input_tick if input_tick is not None else input_mark)
        self.space = self.clamp(self.space + mark - tick)

        return self.space, mark, tick

    def reset(self):
        self.space = self.init_space.clone()

    def to(self, *args, **kwargs):
        self = BaseLayer.to(self, *args, **kwargs)
        
        self.space = self.space.to(*args, **kwargs)
        self.init_space = self.init_space.to(*args, **kwargs)
        self.mark_net = self.mark_net.to(*args, **kwargs)
        self.tick_net = self.tick_net.to(*args, **kwargs)

        return self

class SRNN_Encoder(BaseLayer):
    def __init__(self, input, output, **kwargs):
        BaseLayer.__init__(self, input, output, **kwargs)
        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else 0
        self.hidden_dim = kwargs["hidden_dim"] if "hidden_dim" in kwargs else 30
        self.stig_dim = output

        self.stigmem = StigmergicMemoryLayer_encoder(
            input + self.stig_dim,
            self.stig_dim,
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim
        )

        self.normalization_layer_mark = torch.nn.Linear(self.stig_dim, self.stig_dim)
        self.normalization_layer_tick = torch.nn.Linear(self.stig_dim, self.stig_dim)

        self.init_recurrent = torch.ones(1, self.stig_dim)
        self.reset()
    
    def forward(self, input):

        self.out, mark, tick = self.stigmem(
            torch.cat(
                (input, self.normalization_layer_mark(self.recurrent.expand(input.shape[0],  self.stig_dim)))
            ,1),
            torch.cat(
                (input, self.normalization_layer_tick(self.recurrent.expand(input.shape[0],  self.stig_dim)))
            ,1),
        )
        
        return self.out, mark, tick
    
    def reset(self):
        self.recurrent = self.init_recurrent.clone()
        self.stigmem.reset()

    def to(self, *args, **kwargs):
        self = BaseLayer.to(self, *args, **kwargs)
        
        self.stigmem = self.stigmem.to(*args, **kwargs)
        self.normalization_layer_mark = self.normalization_layer_mark.to(*args, **kwargs)
        self.normalization_layer_tick = self.normalization_layer_tick.to(*args, **kwargs)

        self.init_recurrent = self.init_recurrent.to(*args, **kwargs)
        self.recurrent = self.recurrent.to(*args, **kwargs)
        return self
    
    
class StigmergicMemoryLayer_decoder(BaseLayer):
    def __init__(self, inputs, space_dim, **kwargs):
        BaseLayer.__init__(self, inputs, space_dim)

        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else 0
        self.hidden_dim = kwargs["hidden_dim"] if "hidden_layers" in kwargs else None

        self.n_inputs = inputs
        self.space_dim = self.n_outputs

        self.init_space = torch.zeros(self.space_dim)

        self.mark_net = torch.nn.Sequential()
        self.tick_net = torch.nn.Sequential()

        if self.hidden_layers != 0:
            self.mark_net.add_module("input_w", torch.nn.Linear(self.n_inputs, self.hidden_dim))
            self.tick_net.add_module("input_w", torch.nn.Linear(self.n_inputs, self.hidden_dim))
            
            self.mark_net.add_module("input_s", torch.nn.PReLU())
            self.tick_net.add_module("input_s", torch.nn.PReLU())
            
            for i in range(0, self.hidden_layers-1):
                self.mark_net.add_module("l"+str(i)+"_w", torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                self.tick_net.add_module("l"+str(i)+"_w", torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                
                self.mark_net.add_module("l"+str(i)+"_s", torch.nn.PReLU())
                self.tick_net.add_module("l"+str(i)+"_s", torch.nn.PReLU())
            
            self.mark_net.add_module("output_w", torch.nn.Linear(self.hidden_dim, self.space_dim))
            self.tick_net.add_module("output_w", torch.nn.Linear(self.hidden_dim, self.space_dim))

            self.mark_net.add_module("output_relu", torch.nn.PReLU())
            self.tick_net.add_module("output_relu", torch.nn.PReLU())
        else:
            self.mark_net.add_module("linear", torch.nn.Linear(self.n_inputs, self.space_dim))
            self.tick_net.add_module("linear", torch.nn.Linear(self.n_inputs, self.space_dim))

            self.mark_net.add_module("output_relu", torch.nn.PReLU())
            self.tick_net.add_module("output_relu", torch.nn.PReLU())
            
        self.reset()


    def forward(self, input_mark, input_tick = None):

        mark = self.mark_net(input_mark)
        tick = self.tick_net(input_tick if input_tick is not None else input_mark)
        self.space = self.clamp(self.space + mark - tick)
        
        return self.space, mark, tick

    def reset(self):
        self.space = self.init_space.clone()

    def to(self, *args, **kwargs):
        self = BaseLayer.to(self, *args, **kwargs)
        
        self.space = self.space.to(*args, **kwargs)
        self.init_space = self.init_space.to(*args, **kwargs)
        self.mark_net = self.mark_net.to(*args, **kwargs)
        self.tick_net = self.tick_net.to(*args, **kwargs)

        return self

class SRNN_Decoder(BaseLayer):
    def __init__(self, input, output, **kwargs):
        BaseLayer.__init__(self, input, output, **kwargs)
        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else 0
        self.hidden_dim = kwargs["hidden_dim"] if "hidden_dim" in kwargs else 30
        self.stig_dim = output

        self.stigmem = StigmergicMemoryLayer_decoder(
            input + self.stig_dim,
            self.stig_dim,
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim
        )

        self.normalization_layer_mark = torch.nn.Linear(self.stig_dim, self.stig_dim)
        self.normalization_layer_tick = torch.nn.Linear(self.stig_dim, self.stig_dim)

        self.init_recurrent = torch.zeros(1, self.stig_dim)
        self.reset()
    
    def forward(self, deposit, removal):

        self.out, mark, tick = self.stigmem(
            torch.cat(
                (deposit, self.normalization_layer_mark(self.recurrent.expand(deposit.shape[0], self.stig_dim)))
            ,1),
            torch.cat(
                (removal, self.normalization_layer_tick(self.recurrent.expand(removal.shape[0], self.stig_dim)))
            ,1),
        )
        
        return self.out, mark, tick
    
    def reset(self):
        self.recurrent = self.init_recurrent.clone()
        self.stigmem.reset()

    def to(self, *args, **kwargs):
        self = BaseLayer.to(self, *args, **kwargs)
        
        self.stigmem = self.stigmem.to(*args, **kwargs)
        self.normalization_layer_mark = self.normalization_layer_mark.to(*args, **kwargs)
        self.normalization_layer_tick = self.normalization_layer_tick.to(*args, **kwargs)

        self.init_recurrent = self.init_recurrent.to(*args, **kwargs)
        self.recurrent = self.recurrent.to(*args, **kwargs)
        return self
    
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.Recurrent = SRNN_Encoder(input_size, output_size, hidden_layers = 2 , hidden_dim = hidden_size).to(device)

    def forward(self, input_enc):
        
        #input_enc = input_enc.detach()
        output, mark, tick = self.Recurrent(input_enc)
        return output, mark, tick

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def reset(self):
        self.Recurrent.reset()
        
        
class DecoderRNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.Recurrent = SRNN_Decoder(input_size, hidden_size, hidden_size = input_size, hidden_layers= 2 , hidden_dim = hidden_size).to(device)
        self.out = nn.Linear(hidden_size, 10)
        self.activacion_1 = torch.nn.PReLU() #exp_decay()
        self.out2 = torch.nn.Linear(10, output_size)
        
    def forward(self, deposit, removal):
        deposit = deposit.clone().detach().requires_grad_(False)
        removal = removal.clone().detach().requires_grad_(False)
        #print('mark:', deposit)
        #print('tick:', removal)

        #deposit = F.relu(deposit)
        #removal = F.relu(removal)
        output, mark, tick = self.Recurrent(deposit, removal)
        output = self.out(output)
        output = self.activacion_1(output)
        output = self.out2(output)
        
        return output, mark, tick

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def reset(self):
        self.Recurrent.reset()

