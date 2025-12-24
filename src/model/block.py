#in the trasnformer block we will also add layer normalisation and residual connections
# during backward pass while computing loss, if our outputs are without normalisation we keep getting a chain of multiplications with gradients flowing in between them. These gradients will explode if we don't normalise the values of activatiosn at each layer(outputs at each layer) 
# if gradients are large, even if learning rate is fractional, the change in weight for updation while learning will be huge, so we mightn ot be able to reach the local minima
# to solve this we will need to use even more finegrained activations

#layernorm will just reduce the variance of the activations at each layer so that the gradients are not too large or too small. NO matter how large the activations get layernorm will squish them back to unit scale 

#now when to apply this norm? orignal transformer applied it post attention and shortcut or post mlp plus residual
# loss flows back to layernorm then residual then arrives at atention. The problem is that gradient must pass through layer norm before reaching resudual
# what if layernorms gradient is unstable? required to keep a gradaully increasing learning rate.  (warmup)

#pre norm is gpt 2 onwards where we did x = x + attention(layernorm(x))
#the input to attention is first being layernormed 
#loss-> gradient hits residual first, and then splits into two paths. The other path then has gradients flowing through layer norm and then mlp attention
#one gradient path can directly go to input. Atleatsone path has gradient =1 , no scaling no distortion. Even if blocks aren't trained gradients still flow 
#can use a higher learning rate without warmup 

import torch
import torch.nn as nn
from .config import GPTConfig
from .attention import CausalSelfAttention
from .mlp import MLP 

class TransformerBlock(nn.Module):
    """
    Single Transformer block combining attention and MLP
    Architecture is prenorm 
    x -> layer norm -> attention + residual -> layer norm -> mlp + residual -> output
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
    
    def forward(self, x):
        """
        Wahtever input we get now we will feed it through the blocks we have 
        We get a batch,seq_len,d_model where sequence length is the number of tokens in the sequence and d_model is the dimension of the embedding for each token
        Returns the same thing 
        We will think of it as a stream of input that flows through the network
        """

        #output = shortcut connection(x) + attention(x) (but now we need to prenorm as well that means)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x




