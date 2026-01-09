"""
Testing if our instrumenter captures attention weights properly
"""

import torch
from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.data.tokenizer import Tokenizer 
from src.interpretability.instrumenter import Instrumenter

def main():
    print("-"*50)
    print("Testing instrumentation")
    print("="*50)

    config = GPTConfig()
    model = GPT(config)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    checkpoint = torch.load("checkpoints/epoch_2.pt", map_location="mps")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model is loaded: {sum(p.numel() for p in model.parameters())} parameters/active weights")

    print("\n setting up instrumentation")
    instrumenter = Instrumenter(model)
    instrumenter.register_attention_hooks()
    print(f"registered {len(instrumenter.hooks)} attention hooks, one per layer")


    tokenizer = Tokenizer()
    prompt = "once upon a time"
    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor(tokens, dtype = torch.long).unsqueeze(0).to(device)

    print(f"prompt: {prompt}")
    print(f"tokens: {tokens}")
    print(f"input shape: {input_tensor.shape}")

    print("lets run the forward pass")

    with torch.no_grad():
        output = model(input_tensor)
    
    #single orward pass complete
    print("forward pass complete")
    print(f"output shape: {output.shape}")

    #model will return logits for the input plus the outputoken

    print("the captured attention weights to get to the output of this forward pass")

    for layer_idx in range(len(instrumenter.hooks)):
        attn_weights = instrumenter.get_attention_weights(layer_idx)

        if attn_weights is not None:
            print(f"\n Layer {layer_idx}")
            print(f" Shape of attention weights: {attn_weights.shape}")
            print(f" Expected shape: (batch=1, n_heads=6, seq_len=4, seq_len=4)")

            #lets also check if the attention weights actually act as a proablity distribution or not
            #after softmaxing each row of the attention weights which acts as the row token's attention weight all the other tokens(column). This whole row should be a probablity distribution sho should sum to 1

            sum_check = torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)))
            print(f"attention weights sum to 1 in each row? {sum_check} ")

            print(f"head 0 attention pattern: ")
            print(f" Each row = one tokens attention to all other tokens")

            head_0 = attn_weights[0,0]
            for i, token in enumerate(tokens):
                print(f"Tok{i:>2}", end="  ")
            print()
            
            for i, row in enumerate(head_0):
                print(f"   Tok{i}: ", end="")
                for val in row:
                    print(f"{val.item():5.3f} ", end="")
                print()

        else:
            print(f" layer {layer_idx} no attention weights cpatured")
    
    print("\n" + "-"*50)
    print("comparing heads in layer 0")
    print("-"*50)
    

    layer_0_weights = instrumenter.get_attention_weights(0)
    if layer_0_weights is not None:
        print(f" Layer 0 has {config.n_heads} heads")
        print("to see if they have different patterns")

        for head_idx in range(config.n_heads):
            head = layer_0_weights[0, head_idx]

            #for every head in the layer we will calculate self attnetion and other atteniotn. 
            #self attention is the atention scores/weights for the token in question with its own self that would be the diagonal of the attention matrix
            #other attention is off diagonal head[i,j] where i != j
            #a lower self attention score means that the atention head is social attends to others more than itself, if atteniton score is high it means its introspective 

            self_attention = head.diagonal().mean().item()
            other_attention = (head.sum() - head.diagonal().sum()).item() / (head.numel() - head.size(0))

            print(f"Head {head_idx}:")
            print(f"  Avg self-attention:  {self_attention:.3f}")
            print(f"  Avg other-attention: {other_attention:.3f}")
        
    instrumenter.remove_hooks()
    print("\n instrumentation test complete")


if __name__ == "__main__":
    main()
            



        


            







        

    










