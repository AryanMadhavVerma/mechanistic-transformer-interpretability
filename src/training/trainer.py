"""
Training loop for the model
WE handle forward and backward pass here, loss compute, optimisation, checkpointing and logging
"""

import logging
import torch 
import torch.nn.functional as F 
from torch.optim import AdamW
from tqdm import tqdm
import os
import glob
import re


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        learning_rate=3e-4,
        weight_decay=0.1,
        device="mps",
        checkpoint_dir="checkpoints",
    ):
        """
        We will initialise trainer
        learning_rate: initial learning rate where 3e-4 is a safe one for GPT
        weight_decay: L2 regularisation to prevent overfitting think as gravity pulling the weight towards zero after learning rate helps the weight jump aroudn the curve to find zero loss poin
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        #we initialise the optimiser now 
        self.optimiser = AdamW(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
        )

        self.current_epoch = 0 
        self.global_step = 0
    
    def compute_loss(self, batch):
        """
        We will compute cross entropy loss which is basically softmaxing the logits to create probablities and choose the probablity of the trarget token, and then negative log value of it which is loss
        """
        input = batch[:, :-1]
        target = batch[:, 1:]
        logits = self.model(input)
        #logits shape is batch, seq_len, vocab_size
        #input and target shape is batch, seq_len

        logits_flattened = logits.contiguous().view(-1, logits.size(-1))
        targets_flattened = target.contiguous().view(-1)

        loss = F.cross_entropy(logits_flattened, targets_flattened)

        return loss

    def train_epoch(self):
        """
        Train for one epoch
        """
        #lets set the model to training mode (we enable dropout)
        self.model.train()
        
        total_loss = 0
        num_batches = len(self.train_dataloader)

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            batch = batch.to(self.device)

            loss = self.compute_loss(batch)
            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            total_loss += loss.item()
            self.global_step +=1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        if self.val_dataloader is None:
            return None
        
        self.model.eval() #disables dropout
        total_loss = 0

        with torch.no_grad(): #we dont compute gradients while validating we just use model infference
            for batch in self.val_dataloader:
                batch = batch.to(self.device)
                loss = self.compute_loss(batch)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        return avg_loss
    
    def train(self, num_epochs, resume=True):
        """
        Do multiple iteraitons of the train epoch function based on the num_epoch value
        """

        start_epoch = 0
        if resume:
            checkpoint_loaded = self.load_checkpoint()
            if checkpoint_loaded:
                start_epoch = self.current_epoch + 1  # Continue from next epoch
                print(f"Resuming training from epoch {start_epoch}")
        
        # Check if already completed
        if start_epoch >= num_epochs:
            print(f"Training already completed {start_epoch} epochs (target: {num_epochs})")
            return

        print(f"We are starting training for {num_epochs} epochs (from epoch {start_epoch})")
        print(f"Number of batches we are training: {len(self.train_dataloader)}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch 

            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Training Loss = {train_loss:.4f}")

            if self.val_dataloader:
                val_loss = self.validate()
                print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")

            self.save_checkpoint(f"epoch_{epoch}.pt")
        
        print("Training complete")
    
    def save_checkpoint(self, filename):
        """save model checkpoint"""
        checkpoint = {
            "epoch":self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
        }

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint file in checkpoint directory"""
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "epoch_*.pt")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            return None
        
        epoch_numbers = []
        for ckpt in checkpoints:
            match = re.search(r'epoch_(\d+)\.pt', ckpt)
            if match:
                epoch_numbers.append((int(match.group(1)), ckpt))
        
        if not epoch_numbers:
            return None
        
        # Sort by epoch number and return the latest
        epoch_numbers.sort(key=lambda x: x[0], reverse=True)
        latest_epoch, latest_path = epoch_numbers[0]
        
        return latest_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """
        Load checkpoint to resume training
        If checkpoint_path is None, automatically finds the latest checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is None:
            print("No checkpoint found. Starting training from scratch.")
            return False
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return False
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        
        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        
        print(f"Resumed from epoch {self.current_epoch}, global step {self.global_step}")
        return True
        
        













        




    