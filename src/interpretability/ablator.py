"""
Ablator engine for testing component importance
"""

import torch
from contextlib import contextmanager
from src.model.transformer import GPT

class Ablator:
    """
    We surgically remove attentionweight matrixc of a particular layuer's head and see if it has any impact on loss
    """

    def __init__(self,model: GPT):
        """
        model: GPT Model
        """
        self.model = model
    
    @contextmanager
    def ablate_head(self, layer_idx:int, head_idx:int):
        """
        Context manager to temporarily zero out a specific attention head
        """
        attention = self.model.blocks[layer_idx].attention
        attention.ablated_heads.add(head_idx)

        try:
            yield
        finally:
            attention.ablated_heads.discard(head_idx)
    
       

        





    