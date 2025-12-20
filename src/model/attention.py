import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import GPTConfig

class CausalSelfAttention(nn.Module):
    """
    Multi head causal self attention
    Causal means it an oly attent to past and present toen's vector representation not future
    Self means that it will attend to the same sequence of tokens
    """

    def __init__(self, config:GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, f"d_model {config.d_model} must be divisible by n_heads {config.n_heads}"
        self.W_Q = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_K = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.W_V = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        #output projection will be linear layer with concatenated heads
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        #creating a causal mask so that we mask all tokens for a particular token which come after it in the sequence
        #createa matrix of lower diagonal ones for each attentin matrix
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.context_length, config.context_length)))

        self.dropout = nn.Dropout(config.dropout)

        self.n_heads = config.n_heads
        self.d_head = config.d_head

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """

        B,T,C = x.shape

        # we apply linear projections
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # we now split these into multiple heads 
        # b,t,384 -> b,6,t,64

        # we will first split 384 into 6 and 64
        Q = Q.view(B, T, self.n_heads, self.d_head)
        K = K.view(B, T, self.n_heads, self.d_head)
        V = V.view(B, T, self.n_heads, self.d_head)

        #transfor to B, n_heads, T, d_head
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        # we now compute attention scores

        # Q cross product K(transpose) to get attention score matrix where we get the distance between each token to each other
        scores = Q @ K.transpose(-2,-1) #(this will give us batch, numbero fheads, context_length,context_length which is T)

        #lets also scale the scaled mulopleicaiton to prevent softmax saturation 

        scores = scores / math.sqrt(self.d_head)

        #lets apply a causal mask now 

        scores = scores.masked_fill(self.causal_mask[:T,:T] == 0, float('-inf'))

        #e^ - inf would give 0 so we are essentialyh ignorign these values 

        # lets aply softmax to get weights instead of scores (each row sums to 1 basically so we are making it probablistic) 

        attention_weights = F.softmax(scores, dim=-1)

        #lets apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)
        #now we basically project the atention weights with value matrix to get output 
        #this is conceptually a weighted sum of the value vectors where the weights are the attention weights the output is in the form of B, n_heads, T, d_head
        output = attention_weights @ V 

        #we now reassemble head where we go back from 8,6,10,64 to 8,10,384

        output = output.transpose(1,2)

        output = output.contiguous().view(B, T, C)

        #last thing is to apply the output projection wwhihc in this case is in the same dimeniosn

        output = self.W_O(output)

        output = self.dropout(output)

        return output








