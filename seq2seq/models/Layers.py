import torch.nn as nn
import torch
from models.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from models.EncDecStig import SRNN_Encoder

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
    
    
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, tipo = 'estigmergico'):
        #The argument 'tipo' defines the model architecture it's going to be used
        #Architectures available:
            #'rnn-enc-dec'
            #'stigmergic-enc-dec'
            # 'lstm-enc-dec'
            # 'Transformer'
            # 'stigmergic-output'
            # 'stigmergic-environment'
            # 'stigmergic-total'
        super(EncoderLayer, self).__init__()
        self.tipo = tipo
        if self.tipo != 'stigmergic-enc-dec' and self.tipo != 'rnn-enc-dec'  and self.tipo != 'lstm-enc-dec':
            self.slf_attn = MultiHeadAttention(
                n_head, d_model, d_k, d_v, dropout=dropout)
        
        if self.tipo == 'Transformer':
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        if self.tipo == 'stigmergic-environment' or self.tipo == 'stigmergic-total' or self.tipo == 'stigmergic-output' or self.tipo == 'stigmergic-enc-dec':
            self.SRNN = SRNN_Encoder(d_model, 512).to(device)
            
        if self.tipo == 'rnn-enc-dec':
            self.RNN = torch.nn.RNN(d_model, 512).to(device)
                        
                        
        if self.tipo == 'lstm-enc-dec':
            self.RNN = torch.nn.LSTM(d_model, 512).to(device)
            
        self.enc_output = torch.zeros(160, 512, device=device)
        
        

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        
        if self.tipo == 'rnn-enc-dec':
            encoder_output, hidden = self.RNN(enc_input)   
            return hidden, hidden
        
        if self.tipo == 'lstm-enc-dec':
            encoder_output,(hidden, cn) = self.RNN(enc_input)
            return hidden, hidden
        
        if self.tipo == 'stigmergic-output':
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask)
            enc_output *= non_pad_mask
            
            output_list = []
            for seq in range(enc_output.shape[1]):
                encoder_output, mark, tick = self.SRNN(enc_output[:,seq])
                
                output_list.append(encoder_output)
                self.SRNN.reset()
                
            outputs = torch.stack(output_list, dim = 1)
            

            outputs *= non_pad_mask
            
            return outputs, enc_slf_attn

    
        if self.tipo == 'stigmergic-environment' or self.tipo == 'stigmergic-total':
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask)
            enc_output *= non_pad_mask
            
            balance_list = []
            for seq in range(enc_output.shape[1]):
                encoder_output, mark, tick = self.SRNN(enc_output[:,seq])
                balance = mark - tick
                balance_list.append(balance)
                self.SRNN.reset()
                
            environment = torch.stack(balance_list, dim = 1)
            

            environment *= non_pad_mask
            
            return environment, enc_slf_attn
        
        if self.tipo == 'stigmergic-enc-dec':
            balance_list = []
            for seq in range(enc_input.shape[1]):
                encoder_output, mark, tick = self.SRNN(enc_input[:,seq])#.detach()
                balance = mark - tick
                balance_list.append(balance)
                self.SRNN.reset()
                
            environment = torch.stack(balance_list, dim = 1)
            return environment, environment
    
        if self.tipo == 'Transformer':
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask)
            enc_output *= non_pad_mask

            enc_output = self.pos_ffn(enc_output)
            enc_output *= non_pad_mask
            return enc_output, enc_slf_attn
        
    def reset(self):
        self.SRNN.reset()


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, tipo = 'estigmergico'):
        super(DecoderLayer, self).__init__()
        self.tipo = tipo
        if self.tipo != 'stigmergic-enc-dec' and self.tipo != 'rnn-enc-dec' and self.tipo != 'lstm-enc-dec':
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        if self.tipo == 'Transformer' or self.tipo == 'stigmergic-environment' or self.tipo == 'stigmergic-output':
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        if self.tipo == 'stigmergic-environment' or self.tipo == 'stigmergic-total' or self.tipo == 'stigmergic-output' or self.tipo == 'stigmergic-enc-dec':
            self.SRNN = SRNN_Encoder(d_model, 512).to(device)
            
        if self.tipo == 'rnn-enc-dec':
            self.RNN = torch.nn.RNN(d_model, 512).to(device)
            
            
        if self.tipo == 'lstm-enc-dec':
            self.RNN = torch.nn.LSTM(d_model, 512).to(device)
            
        self.dec_output = torch.zeros(10, 3, 512, device=device)
        

            
        

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        if self.tipo == 'rnn-enc-dec':            
            decoder_output, hidden = self.RNN(dec_input)
            return decoder_output, decoder_output, decoder_output
        
        if self.tipo == 'lstm-enc-dec': 
            encoder_output,(hidden, cn) = self.RNN(enc_input)
            return decoder_output, decoder_output, decoder_output
        
        
        if self.tipo == 'Transformer' or self.tipo == 'stigmergic-environment' or self.tipo == 'stigmergic-output':
            dec_output, dec_slf_attn = self.slf_attn(
                dec_input, dec_input, dec_input, mask=slf_attn_mask)
            dec_output *= non_pad_mask

            dec_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
            dec_output *= non_pad_mask

            dec_output = self.pos_ffn(dec_output)
            dec_output *= non_pad_mask
            
            return dec_output, dec_slf_attn, dec_enc_attn
    
        if self.tipo == 'stigmergic-total':
            
            dec_output, dec_slf_attn = self.slf_attn(
                dec_input, dec_input, dec_input, mask=slf_attn_mask)
            dec_output *= non_pad_mask

            dec_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
            dec_output *= non_pad_mask
            
            balance_list = []
            for seq in range(dec_output.shape[1]):
                decoder_output, mark, tick = self.SRNN(dec_output[:,seq])

                self.SRNN.reset()
                balance_list.append(decoder_output)
                       
            environment = torch.stack(balance_list, dim = 1)
            environment *= non_pad_mask
            return environment, dec_slf_attn, dec_enc_attn
            
        if self.tipo == 'stigmergic-enc-dec':
                output_list = []
                for seq in range(dec_input.shape[1]):
                    decoder_output, mark, tick = self.SRNN(dec_input[:,seq])
                    self.SRNN.reset()
                    output_list.append(decoder_output)

                output = torch.stack(output_list, dim = 1)
                return output, output, output


