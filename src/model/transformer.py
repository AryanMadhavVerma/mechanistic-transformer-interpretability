import torch
import torch.nn as nn
from .config import GPTConfig
from .embedding import Embedding
from .block import TransformerBlock

class GPT(nn.Module):
    """
    Full gpt model: embeddings + transformer blocks + logits generation
    Arch: token_ids -> embeddings -> blocks -> final layern norm -> logits
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embedding= Embedding(self.config)
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.ln_f = nn.LayerNorm(self.config.d_model)
        #now we will create output projection we basically need to trasnformer d_mole into vocab size

        self.W_O = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        #we also need to do weight tying 
        #this means that the output projection weights are the same as the token embedding weights
        #this is a common technique to save memory
        #whenone updates the other updates too
        self.W_O.weight = self.embedding.token_embedding.weight
        #note that we trasnpose w_o later during output calculation during forwar so right now its the same as tokem embedding weights 
        #nn.Linear already takes care of the transposing during forward pass


        #note
        #nn.Linear parameters are always (input_size, output_size). 
        #Pytorch will alwyas transpose you always just write the input and outputsizes 

    def forward(self, x):
        """
        We get the token_ids: batch, seq_len
        we treturn logits: batch, seq_len, vocab_size
        """

        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)

        logits = self.W_O(x)
        return logits   





        





    


