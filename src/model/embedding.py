import torch
import torch.nn as nn
from .config import GPTConfig

class Embedding(nn.Module):
    """
    We will convert token ids intovectors and add positional information as wel
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Embedding(config.context_length, config.d_model)
        #now we will also add dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids):
        """
        we accept the token_ids in the form of (batch,seq_len) - integer token IDs
        and return the embeddings in the form of (batch,seq_len,d_model)
        """

        #we will get the token_embeddings from the input sequence
        token_embeddings = self.token_embedding(token_ids)
        #we will now get the positional information for a sequence (this works for eevery wsequnce as the positional infomraiton is encoded in the positon or index of the token which remains same across the posits for each batch row )
        positional_embeddings = self.positional_embedding(
            torch.arange(token_ids.size(1), device=token_ids.device)
        )
        #now we add them and apply dropout 
        embeddings = token_embeddings + positional_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
    
    






        

