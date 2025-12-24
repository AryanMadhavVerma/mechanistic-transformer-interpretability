#mlp is to store and trasnfporm features at each position 
#we expand the dimension to create more complex features or transformations 
#remove negative values (soften them) using GELU (add non linearity), the nonlinearity would help with smoother gradients 
#compress the projected features back to original dimension

import torch
import torch.nn as nn
import torch.nn.functional as F 
from .config import GPTConfig

class MLP(nn.Module):
    """
    Feed forward network block
    We expand the hidden state(dimensiosn of sequence embedding) to a higher dimension and then contract it back to the original dimension
    We process each token independently(each row of the vectorized batch)
    """

    def __init__(self, config:GPTConfig):
        super().__init__()
        #lets initial two layers where one expands and one contracts (up and down rojections)
        self.config = config
        self.W_1 = nn.Linear(self.config.d_model, self.config.d_ff, bias = self.config.bias)
        self.W_2 = nn.Linear(self.config.d_ff, self.config.d_model, bias = self.config.bias)

        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len_ d_model)
        Returns: 
            output: (batch, seq_len, d_model)
        """

        # we will forst expand the feature set of the input data token by token, sequence by sequence
        x = self.W_1(x)
        # we will now non linearize and activate the weights using GELU
        x = F.gelu(x)

        x = self.W_2(x)

        x = self.dropout(x)

        return x

    
