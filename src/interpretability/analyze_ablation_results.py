"""
Analyze and visualize ablation experiment results
"""
import torch
import torch.nn.functional as F
from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.data.tokenizer import Tokenizer
from src.interpretability.ablator import Ablator
import numpy as np


def compute_loss(model, input_ids, device):
    """Compute cross-entropy loss for next-token prediction"""
    input_ids = input_ids.to(device)
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss


def run_full_ablation_study(model, input_ids, device, config):
    """Run ablation for all heads and return importance matrix"""
    ablator = Ablator(model)
    
    # Compute baseline
    with torch.no_grad():
        baseline_loss = compute_loss(model, input_ids, device)
    
    # Test all heads
    importance_matrix = []
    
    for layer_idx in range(config.n_layers):
        layer_importance = []
        
        for head_idx in range(config.n_heads):
            with torch.no_grad():
                with ablator.ablate_head(layer_idx=layer_idx, head_idx=head_idx):
                    ablated_loss = compute_loss(model, input_ids, device)
            
            delta = (ablated_loss - baseline_loss).item()
            layer_importance.append(delta)
        
        importance_matrix.append(layer_importance)
    
    return baseline_loss.item(), np.array(importance_matrix)


def print_heatmap(importance_matrix, config):
    """Print ASCII heatmap of importance"""
    print("\n" + "="*70)
    print("IMPORTANCE HEATMAP (Δ loss when head is ablated)")
    print("="*70)
    print("Positive = head is important (loss increases when removed)")
    print("Negative = head is harmful (loss decreases when removed)")
    print()
    
    # Header
    print("Layer  ", end="")
    for h in range(config.n_heads):
        print(f"  Head{h}", end="")
    print("  | Avg")
    print("-" * 70)
    
    # Rows
    for layer_idx, row in enumerate(importance_matrix):
        print(f"  {layer_idx}    ", end="")
        for val in row:
            # Color coding with spacing
            if val > 0.08:
                print(f" \033[91m{val:+.3f}\033[0m", end="")  # Red = very important
            elif val > 0.03:
                print(f" \033[93m{val:+.3f}\033[0m", end="")  # Yellow = important
            elif val < -0.03:
                print(f" \033[92m{val:+.3f}\033[0m", end="")  # Green = harmful
            else:
                print(f" {val:+.3f}", end="")  # White = neutral
        
        avg = np.mean(row)
        print(f" | {avg:+.3f}")
    
    print("-" * 70)
    
    # Column averages
    print("Avg    ", end="")
    for col_idx in range(config.n_heads):
        col_avg = np.mean(importance_matrix[:, col_idx])
        print(f" {col_avg:+.3f}", end="")
    print(f" | {np.mean(importance_matrix):+.3f}")
    print()


def print_rankings(importance_matrix, config):
    """Print ranked lists of heads"""
    print("head importance matrix")
    # Flatten and create (layer, head, delta) tuples
    head_list = []
    for layer_idx in range(config.n_layers):
        for head_idx in range(config.n_heads):
            delta = importance_matrix[layer_idx, head_idx]
            head_list.append((layer_idx, head_idx, delta))
    
    # Sort by delta (descending)
    head_list.sort(key=lambda x: x[2], reverse=True)
    
    print("\n 10 most imp heads (removing hurts most):")
    for rank, (layer, head, delta) in enumerate(head_list[:10], 1):
        print(f"{rank:2d}. Layer {layer}, Head {head}: Δ = {delta:+.4f}")
    
    print("\n10 most detrimental heads (removing helps!):")
    for rank, (layer, head, delta) in enumerate(reversed(head_list[-10:]), 1):
        print(f"{rank:2d}. Layer {layer}, Head {head}: Δ = {delta:+.4f}")


def print_layer_summary(importance_matrix, config):
    """Summarize importance by layer"""
    print("layer level summary")
    
    for layer_idx in range(config.n_layers):
        layer_deltas = importance_matrix[layer_idx]
        
        mean_delta = np.mean(layer_deltas)
        max_delta = np.max(layer_deltas)
        min_delta = np.min(layer_deltas)
        std_delta = np.std(layer_deltas)
        
        # Count head types
        important_heads = np.sum(layer_deltas > 0.03)
        harmful_heads = np.sum(layer_deltas < -0.03)
        neutral_heads = config.n_heads - important_heads - harmful_heads
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Mean Δ: {mean_delta:+.4f}  |  Range: [{min_delta:+.4f}, {max_delta:+.4f}]  |  Std: {std_delta:.4f}")
        print(f"  Heads: {important_heads} important, {neutral_heads} neutral, {harmful_heads} harmful")


def main():
    print("-"*30)
    print("visualising ablations")
    
    # Setup
    config = GPTConfig()
    model = GPT(config)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load("checkpoints/epoch_2.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")
    
    # Create test input
    tokenizer = Tokenizer()
    prompt = "once upon a time there was a"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    print(f"\nTest prompt: '{prompt}'")
    print(f"Tokens: {tokens}")
    print(f"Sequence length: {len(tokens)}")
    
    # Run ablation study
    print("\nRunning full ablation study (36 tests)...")
    baseline_loss, importance_matrix = run_full_ablation_study(model, input_ids, device, config)
    
    print(f"\nBaseline loss: {baseline_loss:.4f}")
    
    # Visualizations
    print_heatmap(importance_matrix, config)
    print_rankings(importance_matrix, config)
    print_layer_summary(importance_matrix, config)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    overall_mean = np.mean(importance_matrix)
    overall_std = np.std(importance_matrix)
    
    print(f"\n1. Average head importance: {overall_mean:+.4f} (std: {overall_std:.4f})")
    print(f"   → Heads show {overall_std/abs(overall_mean + 1e-8):.2f}× variation in importance")
    
    positive_heads = np.sum(importance_matrix > 0)
    negative_heads = np.sum(importance_matrix < 0)
    print(f"\n2. Head distribution: {positive_heads} helpful, {negative_heads} harmful")
    print(f"   → {negative_heads/36*100:.1f}% of heads hurt performance!")
    
    layer_means = np.mean(importance_matrix, axis=1)
    most_important_layer = np.argmax(layer_means)
    print(f"\n3. Most important layer: Layer {most_important_layer} (avg Δ = {layer_means[most_important_layer]:+.4f})")
    print(f"   → Challenges assumption that early layers are most critical")
    
    max_importance = np.max(importance_matrix)
    min_importance = np.min(importance_matrix)
    print(f"\n4. Importance range: {min_importance:+.4f} to {max_importance:+.4f}")
    print(f"   → {max_importance/abs(min_importance):.2f}× difference between best and worst head")
    


if __name__ == "__main__":
    main()
