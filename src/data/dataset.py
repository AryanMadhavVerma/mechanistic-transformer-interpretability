"""
We use tinystory dataset to train our model
we will download data from huggingface, tokenize, chunk into fixed-length sequences
"""

import torch
from torch.utils.data import Dataset, DataLoader, dataloader
from datasets import load_dataset
from .tokenizer import Tokenizer

class TinyStoriesDataset(Dataset):
    """
    Tinystories dataset which we will chunk into context_length sequence despite the lenghto f a single story 
    Each example we create will have max length of context length
    For next token preductions we will use input: tokens[:-1], target:tokens[1:]
    """

    def __init__(
        self,
        split: str = "train",
        context_length: int = 512,
        max_examples: int = None,
    ):
        """
        We will initalise the dataset, split it 
        """

        self.context_length = context_length
        self.tokenizer = Tokenizer()

        #we will now load the dataset from huggingface, which wil download it for the first time before storing in cache
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        
        all_tokens = []
        for i, example in enumerate(self.dataset):
            if max_examples and i >= max_examples:
                break

            story_text = example['text']
            tokens = self.tokenizer.encode(story_text)
            all_tokens.extend(tokens)
        
        #we will have all stories in the form of a stream of tokens which we will chun aferwards
        self.examples = []
        for i in range(0,len(all_tokens) - context_length, context_length):
            chunk = all_tokens[i:i+context_length]
            self.examples.append(chunk)

    def __len__(self):
        """number of training exmples """
        return len(self.examples)
    
    def __getitem__(self, idx):
        """getting one trainig example"""
        tokens = self.examples[idx]
        return torch.tensor(tokens, dtype=torch.long)


def create_dataloader(
    split: str = "train",
    batch_size: int = 8,
    context_length: int = 512,
    max_examples: int = None,
):
    """
    We will create dataloader for training
    We split on train or validation
    batch_size: Number of sequences per batch
    we return a dataloader which gives us batches of shape (batch_size, context_length)
    """
    dataset = TinyStoriesDataset(
        split=split,
        context_length=context_length,
        max_examples=max_examples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),  #we shuffle the training data not the validation data
        num_workers = 0, #playing thread safe for mac compatiblity
    )

    return dataloader
        
        


