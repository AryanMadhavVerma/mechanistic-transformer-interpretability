"""GPT Model configuration"""

from dataclasses import dataclass

@dataclass 
class GPTConfig:
    """Config for the gpt model that we will use
        We wil use the 10M parameter model for tinystories data set training on m2 chip
    """

    vocab_size: int = 50257
    context_length: int = 512

    n_layers: int = 6
    n_heads: int = 6
    d_model: int= 384 #the internal dimensions of hidden layers, also divislbe by number of heads, small enough to inspect unliked to bigger models
    d_ff: int = 1536 #ff dimension is mostly 4 times the internal dimensions, it is a multiple because feature extraction needs capacity for storage of knowledge as well 

    dropout: float = 0.1

    bias: bool = False # no bias in linear layers (input*wT + 0*bias)

    def __post_init__(self):
        """
        lets validate config after initialisation
        """
        assert self.d_model % self.n_heads == 0, f"d_model {self.d_model}must be divisible by n_heads {self.n_heads}"

    @property
    def d_head(self) -> int: 
        """dimensions per attention head"""
        return self.d_model // self.n_heads

    def estimate_params(self) -> int:
        """
        Rough parameter count
        """
        token_emb = self.vocab_size * self.d_model
        pos_emb = self.context_length * self.d_model

        attn = 4 * self.d_model * self.d_model #the key value and query vector matrices with input and output both the same as input dimesnions
        #we project the entire input 384 dims into Q/K/V where here even the output after projection would have same dimension
        #then we will split it for multihead attention
        mlp = self.d_model * self.d_ff + self.d_ff * self.d_model #we have the layer which expnds input space and then contracts it

        ln = 2 * self.d_model #this is basically when we do mean and variance calculation for each layer later. THis has a scale gamme and a shift beta params per dimension
        #scale gets multiplied after normalisation, shift gets added after normalisaiton. We will have to add it to the model later
        # so there will be basically scale and shift across inputs that means they wil just have the dimension clum,n so 384 ifferenct scale and shift vectors 
        # normalised = x- mean / sqrt(var(x) + e)
        # output = gamma * normalised plus beta. these gamma nad beta values are learnable

        per_block = attn + mlp + ln 
        final_ln = self.d_model #(the final nalyer normalisaiton)

        #output projection layer will be the same as the token embedding matrix (weight typing)

        return token_emb + pos_emb + self.n_layers * per_block + final_ln

    

        

