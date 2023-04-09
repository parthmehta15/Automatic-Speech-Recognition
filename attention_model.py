

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

letter2index = {'<sos>': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26, "'": 27, ' ': 28, '<eos>': 29}


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()
    def forward(self, x):
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)

        return x * mask

class LockedDropout2(nn.Module):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()
    def forward(self, x):
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)

        return x * mask


class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm1 = nn.LSTM(input_size=input_dim, hidden_size = hidden_dim, num_layers=1,batch_first =True, bidirectional = True, bias = True)
        self.lock = LockedDropout(p=0.1)
        self.blstm1.apply(self._init_weights)

    def _init_weights(self, m):
        for param in m.parameters():
            nn.init.uniform_(param.data, -0.1, 0.1)

    def forward(self, x):
        x, len_x = pad_packed_sequence(x,batch_first=True)
        if x.shape[1]%2!=0:
          x = x[:,:torch.div(x.shape[1], 2, rounding_mode ='floor')*2, :]
        else:
          pass

        x = x.reshape((x.shape[0], int(x.shape[1]/2),x.shape[2]*2))
        len_x = torch.div(len_x, 2, rounding_mode ='floor')
        packed_input = pack_padded_sequence(x,len_x,batch_first=True, enforce_sorted=False)
        out_a1, (out_a2, out_a3) = self.blstm1(packed_input)
        x, len_x = pad_packed_sequence(out_a1,batch_first=True)
        out_drop = self.lock(x)
        packed_input = pack_padded_sequence(out_drop,len_x,batch_first=True, enforce_sorted=False)

        return packed_input


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size = encoder_hidden_dim, num_layers=1,batch_first =True, bidirectional = True)
        self.lstm.apply(self.initialize_weights)
       
        self.pBLSTMs = nn.Sequential(
            pBLSTM(4*encoder_hidden_dim, encoder_hidden_dim),
            pBLSTM(4*encoder_hidden_dim, encoder_hidden_dim),
            pBLSTM(4*encoder_hidden_dim, encoder_hidden_dim),
            
        )
         
        self.key_network = nn.Linear(2*encoder_hidden_dim, key_value_size)
        self.value_network = nn.Linear(2*encoder_hidden_dim, key_value_size)

    def initialize_weights(self, module):
        for parameters in module.parameters():
            nn.init.uniform_(parameters.data, -0.1, 0.1)

    def forward(self, x, x_len):
        packed_input = pack_padded_sequence(x,x_len,batch_first=True, enforce_sorted=False)
        out1, out2 = self.lstm(packed_input)
        out_new = self.pBLSTMs(out1)
        x, len_x = pad_packed_sequence(out_new,batch_first=True)
        keys = self.key_network(x)
        value_network = self.value_network(x)
        
        return keys, value_network, len_x



class Attention(nn.Module):
 
    def __init__(self, key_value_size):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask):
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        energy.masked_fill_(mask,float("-inf"))
        attention =  F.softmax(energy,dim = 1) 
        context = torch.bmm(attention.unsqueeze(1),value).squeeze(1) 

        return context, attention
    


class Decoder(nn.Module):
    '''
    Each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell can be used as a query for calculating attention.
    '''


    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
       
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_dim, padding_idx = letter2index['<eos>'])
        
        self.lstm1 = nn.LSTMCell(input_size = embed_dim+key_value_size, hidden_size = decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size = decoder_hidden_dim, hidden_size = key_value_size)
        
        self.attention = Attention(key_value_size)     
        
        self.vocab_size = vocab_size
        
        self.lock1 = LockedDropout2(0.2)
        self.lock2 = LockedDropout2(0.2)
        
        self.query_ful = nn.Linear(key_value_size, key_value_size)
        self.context_ful = nn.Linear(key_value_size, key_value_size)
        
        self.embedding.apply(self.initialize_weights)

        self.character_prob = nn.Linear(key_value_size*2, vocab_size)
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight

    def initialize_weights(self, module):
        for parameters in module.parameters():
            nn.init.uniform_(parameters.data, -0.1, 0.1)

    def forward(self, key, value, encoder_len, y, mode, teacher_forcing, teacher_rate):
        '''
        Args:
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            y: (B, text_len) - Batch input of text with text_length
            mode: Train or eval mode for teacher forcing
        Return:
            predictions: the character perdiction probability 

        '''
        B, key_seq_max_len, key_value_size = key.shape
        
        if mode == 'train':
            max_len =  y.shape[1]
            char_embeddings = self.embedding(y) 
        else:
            max_len = 600

        mask = torch.arange(key_seq_max_len).unsqueeze(0) >= encoder_len.unsqueeze(1) 
        mask = mask.to(device)


        
        predictions = []
        prediction =  torch.full((B,), fill_value= 0, device=device)
        
        hidden_states = [None, None] 
        context = value[:,0,:]
        attention_plot = [] 

        for i in range(max_len):
            if mode == 'train':  

                p = np.random.random()
                if teacher_rate >=p:
                  teacher_forcing = True
                else:
                  teacher_forcing = False

                
                if teacher_forcing == True:
                  if i ==0:
                    dummy = torch.full(size = (B,), fill_value = letter2index['<sos>'], dtype=torch.long).cuda()
                    dummy = dummy.fill_(letter2index['<sos>']).to(device)

                    char_embed = self.embedding(dummy)
                    # print('char_embed',char_embed.shape)
                  else:
                    char_embed = char_embeddings[:,i-1,:]
                    # print('char_embed',char_embed.shape)
                else:
                  if i ==0:
                    dummy = torch.full(size = (B,), fill_value = letter2index['<sos>'], dtype=torch.long).cuda()
                    char_embed = self.embedding(dummy)
                  else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))

            else:
                if i ==0:

                  dummy = torch.full(size = (B,), fill_value = letter2index['<sos>'], dtype=torch.long).cuda()
                  char_embed = self.embedding(dummy)
                else:
                  char_embed = self.embedding(prediction.argmax(dim=-1)) 
               
            y_context = torch.cat([char_embed, context], dim=1)

            # context and hidden states of lstm 1 from the previous time step should be fed
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])
            dummy0 = self.lock1(hidden_states[0][0].unsqueeze(0)).squeeze(0)
            hidden_states[0] = (dummy0, hidden_states[0][1])
            

            # hidden states of lstm1 and hidden states of lstm2 from the previous time step should be fed
            hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1])
            dummy1 = self.lock2(hidden_states[1][0].unsqueeze(0)).squeeze(0)
            hidden_states[1] = (dummy1, hidden_states[1][1]) 
            

            
            query = hidden_states[1][0] 
            #query mlp
            query = self.query_ful(query)
            
            # Compute attention from the output of the second LSTM Cell
            context, attention = self.attention(query, key, value, mask)
            # contex mlp 
            context = self.context_ful(context)

            attention_plot.append(attention[0].detach().cpu())
            
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            predictions.append(prediction.unsqueeze(1))# fill this out)

        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)

        return predictions, attentions
    


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, key_value_size)

    def forward(self, x, x_len, y, mode, teacher_forcing, teacher_rate):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode= mode, teacher_forcing = teacher_forcing, teacher_rate = teacher_rate)
        
        return predictions, attentions
    

