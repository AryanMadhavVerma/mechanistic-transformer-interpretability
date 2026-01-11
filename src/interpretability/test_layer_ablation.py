"""
Test ablating entire layers to understand layer-level importance
"""
import torch
import torch.nn.functional as F
from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.data.tokenizer import Tokenizer
from src.interpretability.ablator import Ablator
import numpy as np


def compute_loss(model, input_ids, device):
    """Computing cross-entropy loss"""
    input_ids = input_ids.to(device)
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss


def main():
    print("layer ablations")
    print("-"*30)
    
    # Load model
    config = GPTConfig()
    model = GPT(config)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load("checkpoints/epoch_2.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create test input
    tokenizer = Tokenizer()
    prompt = "once upon a time there was a"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    
    ablator = Ablator(model)
    
    # Baseline
    with torch.no_grad():
        baseline_loss = compute_loss(model, input_ids, device)
    print(f"\nBaseline loss: {baseline_loss:.4f}")
    
    print("Ablating entire layers (all 6 heads in each layer)")
    print("-"*70)
    
    layer_deltas = []
    
    for layer_idx in range(config.n_layers):
        with torch.no_grad():
            with ablator.ablate_layer(layer_idx):
                ablated_loss = compute_loss(model, input_ids, device)
        
        delta = ablated_loss - baseline_loss
        layer_deltas.append(delta.item())
        
        print(f"Layer {layer_idx}: loss = {ablated_loss:.4f}, delta = {delta:+.4f}")
    
    print("layer importance")
    print("-"*30)
    
    layer_ranking = sorted(enumerate(layer_deltas), key=lambda x: x[1], reverse=True)
    
    print("\most imp layers (highest delta when all heads removed):")
    for rank, (layer_idx, delta) in enumerate(layer_ranking, 1):
        print(f"{rank}. Layer {layer_idx}: delta = {delta:+.4f}")
    
    # Group analysis
    print("\n" + "-"*70)
    print("Early vs Middle vs Late layers:")
    print("-"*70)
    
    early = np.mean([abs(layer_deltas[0]), abs(layer_deltas[1])])
    middle = np.mean([abs(layer_deltas[2]), abs(layer_deltas[3])])
    late = np.mean([abs(layer_deltas[4]), abs(layer_deltas[5])])
    
    print(f"Early layers (0-1):  avg delta = {early:.4f}")
    print(f"Middle layers (2-3): avg delta = {middle:.4f}")
    print(f"Late layers (4-5):   avg delta = {late:.4f}")

    
    # comparing with single head ablation finding
    print("layer ablation vs single head ablation:")
    print("-"*70)
    
    # worst single head was Layer 3 Head 3 at +0.1409
    max_single_head_delta = 0.1409
    max_layer_delta = max(layer_deltas)
    
    ratio = max_layer_delta / max_single_head_delta
    print(f"Max single head delta: {max_single_head_delta:.4f}")
    print(f"Max layer delta:       {max_layer_delta:.4f}")
    print(f"Ratio:             {ratio:.2f}x")
    
    

if __name__ == "__main__":
    main()
