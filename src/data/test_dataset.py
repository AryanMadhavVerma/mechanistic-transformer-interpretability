from src.data.dataset import create_dataloader

dataloader = create_dataloader(
    split="train",
    batch_size=4,
    context_length=512,
    max_examples=100
)

print(f"number of batches: {len(dataloader)}")
batch = next(iter(dataloader))
print(f"batch shape: {batch.shape}")
print(f"batch dtype: {batch.dtype}")
print(f"token range: {batch.min()} - {batch.max()}")

from src.data.tokenizer import Tokenizer
tokenizer = Tokenizer()
first_sequence = batch[0].tolist()
decoded_sequence = tokenizer.decode(first_sequence)
print(f"first sequence: {decoded_sequence}")
