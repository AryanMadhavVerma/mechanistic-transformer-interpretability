import torch
import torch.nn.functional as F

from src.model.config import GPTConfig
from src.data.tokenizer import Tokenizer
from src.model.transformer import GPT

class Generator:
    def __init__(self, config: GPTConfig, model: GPT, tokenizer: Tokenizer, device):
        self.config = config
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval() #we will set the model to evaluation mode

    def generate(self, prompt:str, max_tokens=50, temperature=1.0, top_k:int = None):
        """ 
        We will generate text autoregressively which means we will generate one token at a time based on p[revioous token 
        """
        tokenized_prompt = self.tokenizer.encode(prompt)
        print(f"This is what the tokenized prompt looks like: {tokenized_prompt}")
        # we will now conver tthe tokenized text into tensors basically each token will now be spread across the embedding dimension size as a vector
        #after converting to a tensor stream, we will add batch dimension, move to device 
        prompt_tensor = torch.tensor(tokenized_prompt, dtype=torch.long)
        print(f"This is what the prompt tensor looks like: {prompt_tensor}")

        #we need to now add a batch dimension at the first index 
        batched_prompt_tensor = torch.unsqueeze(prompt_tensor, 0).to(device=self.device)


        print(f"This is what the batched proimpt tesnor loks liek aftger applying unsqueeze {batched_prompt_tensor}")
        
        #creating a temporary context state where gradients will not be computed
        with torch.no_grad():
            #we will only loop / generate new tokens till max tokens
            for _ in range(max_tokens):
                #we handle context window limit for hte input, (not greater than 512, trim from the left if it is, keep the most recent 512 tokens) 
                current_seq = batched_prompt_tensor
                if current_seq.size(1) > self.config.context_length:
                    current_seq = current_seq[:, -self.config.context_length:]

                #we get the predictions ([1,seqlen] -> [1,seqlen,vocabsize])
                logits = self.model(current_seq)

                #we extract only the last positions logits
                # we only care about what comes after our current sequence 
                last_logits = logits[:, -1, :]

                #we apply post generation optimisations like temperature nad top k to choose the best next token 
                #temperature we divide by it, so anything above 1 means that we reduce spikes and make the probablities more uniform across the vocabulary 
                scaled_logits = last_logits / temperature

                #top-k filtering: out of all the vocabulary if we dont wnt to hcoose betwee nall 50276 probablities we can reduce to the top k probablities and choose from them

                if top_k is not None:
                    #k-th largets value
                    values, _ = torch.topk(scaled_logits, top_k)
                    #gives us k largest values and their indices in descending order( we dont care about indices right now so we will just use vlaues )
                    min_value = values[:,-1]
                    #because we accessed the element after the first dimension now the output of miun_value has shape (1,) // [number]
                    #but we need it to be in the same shape as the logits for comparision we need to broadcast it 
                    min_value = min_value.unsqueeze(-1)
                    
                    scaled_logits = torch.where(
                        scaled_logits < min_value,
                        torch.tensor(float('-inf')).to(self.device),
                        scaled_logits,
                    )

                
                #now we have the scaled logits (if top k is there we have scaled other than topkj values to -inf so that after softmaxing they go down to 0)

                #we now convert to probablitys (sum = 1)
                probs = F.softmax(scaled_logits, dim=-1)
                
                
                #we sample out the next token 
                #instead of always piocking argamax which  is th e highest probablity token we sample from the distribution randomly 
                next_token = torch.multinomial(probs,num_samples=1)

                #appending new token to input sequence and then concatenating along the squence dimension 
                batched_prompt_tensor = torch.cat([batched_prompt_tensor, next_token], dim=1)
        
        #we now convert the complete tensor with additional new generated tokens back to token id form
        #lets remove the batch 
        generated_ids = batched_prompt_tensor.squeeze(0).tolist() #to list to convert back from tensor (now single dimensioned) to list of integers
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

        


            



        