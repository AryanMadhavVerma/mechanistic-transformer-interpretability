import torch
import torch.nn.functional as F
from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.data.tokenizer import Tokenizer
from src.interpretability.ablator import Ablator




def compute_loss(model, input_ids, device):
    input_ids = input_ids.to(device)
    logits = model(input_ids[:,:-1])
    targets = input_ids[:,1:]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss

def main():
    print("testing ablation engine --")

    config = GPTConfig()
    model = GPT(config)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = torch.load("checkpoints/epoch_4.pt", map_location = device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()


    print(f"model is loaded. params: {sum(p.numel() for p in model.parameters())}")

    # we will now create test input

    tokenizer = Tokenizer()
    
    prompt = "once upon a time there was a"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) #can either use unsqueeze or can directly put as [tokens] inside torch.tensor()

    with torch.no_grad():
        baseline_loss = compute_loss(model, input_ids, device)
        print(f"base line loss calculated is : {baseline_loss:.4f}")

    print("ablating heads in layer 0, lets see")

    ablator = Ablator(model)
    
    for layer_idx in range(config.n_layers):
        for head_idx in range(config.n_heads):
            with torch.no_grad():
                with ablator.ablate_head(layer_idx=layer_idx, head_idx=head_idx):
                    ablated_loss = compute_loss(model, input_ids, device)
            
            delta = ablated_loss - baseline_loss
            print(f"layer {layer_idx}, head {head_idx}: loss = {ablated_loss:.4f}, delta = {delta:.4f}")

    with torch.no_grad():
        final_loss = compute_loss(model, input_ids, device)
    print(f"\nFinal loss (should match baseline): {final_loss:.4f}")
    print(f"Restoration successful: {torch.allclose(baseline_loss, final_loss)}")

if __name__ == "__main__":
    main()

            

    


    

    
    


