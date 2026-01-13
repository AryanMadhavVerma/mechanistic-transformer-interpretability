import torch
from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.data.tokenizer import Tokenizer
from src.generation.generator import Generator


config = GPTConfig()
model = GPT(config)

checkpoint = torch.load("checkpoints/epoch_4.pt",map_location="mps")
model.load_state_dict(checkpoint["model_state_dict"])

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = Tokenizer()
generator = Generator(config, model, tokenizer, device)

prompt = "Once upon a time"
print(f"prompt is: {prompt}")
print("-"*50)

output = generator.generate(prompt, max_tokens=50, temperature=1)
print(output)