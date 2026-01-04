"""
We will initialise the tokenizer for gpt. We will use BPE where subword tokenisation takes place between charactesr and words tilll we reac hthe 50257 token vocabulary
This can handle even made up words
"""

import tiktoken

class Tokenizer:
    """
    Wrapper around the tiktoken tokenizer
    We will have a method called encode to convert text into token id using vocaulry as lookup
    We will have a decode function whic takes the id and converst it back to text will be helpful lateronce we get the logits
    """

    def __init__(self, model_name:str="gpt2"):
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.vocab_size = self.tokenizer.n_vocab

        #lets define some custom end of text token
        self.eot_token = self.tokenizer.eot_token

    def encode(self, text:str) -> list[int]:
        """
        Convert text to token ids
        We get a string and we return with a list of integers as tokens to those word-character comboinatiosn
        """
        return self.tokenizer.encode(text)

    def decode(self, token_ids:list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """
        Encode multiple sequences together
        """
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_ids_list: list[list[int]]) -> list[str]:
        """
        Decode multiple sequences together
        """
        return [self.decode(token_ids) for token_ids in token_ids_list]
        

def test_tokenizer():
    """Test basic tokenization."""
    tokenizer = Tokenizer()
    
    # Test 1: Simple text
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Match: {text == decoded}")
    
    # Test 2: Show tokenization patterns
    examples = [
        "the",           # Common word → single token
        "uncommon",      # Gets split
        "GPT-5000",      # Unknown → splits nicely
        " the",          # Space matters!
    ]
    
    print("\n--- Tokenization Examples ---")
    for ex in examples:
        tokens = tokenizer.encode(ex)
        print(f"{ex:15} → {tokens} ({len(tokens)} tokens)")


if __name__ == "__main__":
    test_tokenizer()