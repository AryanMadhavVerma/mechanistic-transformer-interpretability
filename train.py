"""
We will train on a small subset of TinyStories to verify everything works
"""

import torch
from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.data.dataset import create_dataloader
from src.training.trainer import Trainer

def main():
    config = GPTConfig(
        vocab_size=50257,
        context_length=512,
        n_layers=6,
        n_heads=6,
        d_model=384,
        d_ff=1536,
        dropout=0.1,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device: {device}")
    
    # Model
    print("initialising model")
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())

    print("\n loading data...")
    train_dataloader = create_dataloader(
        split="train",
        batch_size=8,
        context_length=config.context_length,
        max_examples=20000
    )

    val_dataloader = create_dataloader(
        split="validation",
        batch_size=8,
        context_length=config.context_length,
        max_examples=2000
    )

    print(f"length of training and validation batches ius {len(train_dataloader)}")

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=2e-4,
        weight_decay=0.1,
        device=device,
        checkpoint_dir="checkpoints"
    )

    print("\n" + "="*50)
    print("starting training...")
    print("="*50 + "\n")

    # Training will automatically resume from the latest checkpoint if found
    # Set resume=False to start from scratch: trainer.train(num_epochs=10, resume=False)
    trainer.train(num_epochs=10, resume=True)

    print("\n" + "="*50)
    print("training complete")
    print("="*50)


if __name__ == "__main__":
    main()













