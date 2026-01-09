import torch
from typing import Dict, List
from collections import defaultdict

#we are building this so that register hooks on all attention layers, collect activations (intermdiate outputs) during a forward pass, easy access to the collected data

class Instrumenter:
    """
    Captures internal activations from a model during forward pass
    """
    
    def __init__(self, model):
        self.model = model
        self.activations = defaultdict(list) #we will store captured activations
        self.hooks = [] #we will track registered hooks

    def register_attention_hooks(self):
        """
        Register hooks on all attention layers to capture attention weights
        """
        #we will loop through the model blocks and for each block we will register a hook on attention block
        #the hook will capture attention weights
        def make_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(module, 'attention_weights'):
                    key = f"attention_layer_{layer_idx}"
                    self.activations[key].append(module.attention_weights.detach().cpu()) #removes from computation graph which its stored for, for backprop, no gradients get tracked we move to CPU and free up GPU for inference space
            return hook
        
        for i, block in enumerate(self.model.blocks):
            hook_function = make_hook(i)
            hook_handle = block.attention.register_forward_hook(hook_function)
            self.hooks.append(hook_handle)


        

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        """Remove all regsitered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_attention_weights(self, layer_idx: int):
        """
        Get attention weights for a specific layer 
        Returns tesnro of shape (batch, n_heads, seq_len,seq_len)
        """
        key = f"attention_layer_{layer_idx}"
        if key in self.activations:
            return self.activations[key][0]
        return None


        
        