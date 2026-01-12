"""
Ablator engine for testing component importance
"""

import torch
from contextlib import contextmanager
from src.model.transformer import GPT
from typing import Dict, List, Union

class Ablator:
    """
    Surgically ablate attention heads or entire layers
    """

    def __init__(self, model: GPT):
        self.model = model
    
    @contextmanager
    def ablate_head(self, layer_idx: int, head_idx: int):
        """
        Ablate a single attention head
        
        Args:
            layer_idx: Which layer (0-5)
            head_idx: Which head within that layer (0-5)
        """
        attention = self.model.blocks[layer_idx].attention
        attention.ablated_heads.add(head_idx)
        
        try:
            yield
        finally:
            attention.ablated_heads.discard(head_idx)
    
    @contextmanager
    def ablate_heads(self, ablation_map: Dict[int, List[int]]):
        """
        Ablate multiple heads across multiple layers
        
        
        """
        # Marking all heads for ablation
        for layer_idx, head_list in ablation_map.items():
            attention = self.model.blocks[layer_idx].attention
            for head_idx in head_list:
                attention.ablated_heads.add(head_idx)
        
        try:
            yield
        finally:
            # Restore all heads
            for layer_idx, head_list in ablation_map.items():
                attention = self.model.blocks[layer_idx].attention
                for head_idx in head_list:
                    attention.ablated_heads.discard(head_idx)
    
    @contextmanager
    def ablate_layer(self, layer_idx: int):
        """
        Ablate all attention heads in a layer
        
        Args:
            layer_idx: Which layer to ablate (0-5)
        """
        attention = self.model.blocks[layer_idx].attention
        n_heads = len(attention.ablated_heads) if hasattr(attention, 'n_heads') else 6
        
        # Add all head indices
        for head_idx in range(self.model.config.n_heads):
            attention.ablated_heads.add(head_idx)
        
        try:
            yield
        finally:
            attention.ablated_heads.clear()

    
    @contextmanager
    def ablate_mlp(self,layer_idx:int):
        mlp = self.model.blocks[layer_idx].mlp
        mlp.ablated = True
        
        try:
            yield
        finally:
            mlp.ablated = False
    


    