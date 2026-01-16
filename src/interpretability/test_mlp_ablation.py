"""
testing how mlp changes affect the entire decoder gpt network 
to ablate each layer of mlp in the transformer block and seeing how loss is affected

the mlp layers have hiddened states activations it is not a parallel operation like attention but sequentially processes each token
"""
import torch
import torch.nn.functional as F
from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.data.tokenizer import Tokenizer
from src.interpretability.ablator import Ablator
import glob


def compute_loss(model, input_ids, device):
    """compute cross-entropy loss for next token prediction"""
    input_ids = input_ids.to(device)
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss


def main():
    print("mlp ablation study")
    print("-"*30)
    
    # load model
    config = GPTConfig()
    model = GPT(config)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint_files = glob.glob("checkpoints/epoch_*.pt")
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in checkpoints directory")
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    print(f"Loading the latest checkpoint {latest_checkpoint} for analysis")
    checkpoint = torch.load(latest_checkpoint, map_location="mps")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # create test input
    tokenizer = Tokenizer()
    prompt = "once upon a time there was a"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    ablator = Ablator(model)
    
    # baseline
    with torch.no_grad():
        baseline_loss = compute_loss(model, input_ids, device)
    print(f"\nbaseline loss: {baseline_loss:.4f}")
    
    # test each layer's mlp
    print("\nablating mlp blocks")
    print("-"*70)
    
    for layer_idx in range(config.n_layers):
        with torch.no_grad():
            with ablator.ablate_mlp(layer_idx):
                ablated_loss = compute_loss(model, input_ids, device)
        
        delta = ablated_loss - baseline_loss
        print(f"layer {layer_idx} mlp: loss = {ablated_loss:.4f}, delta = {delta:+.4f}")
    
    # restoration check
    with torch.no_grad():
        final_loss = compute_loss(model, input_ids, device)
    
    print(f"\nfinal loss: {final_loss:.4f}")
    print(f"restoration successful: {torch.allclose(baseline_loss, final_loss)}")


if __name__ == "__main__":
    main()