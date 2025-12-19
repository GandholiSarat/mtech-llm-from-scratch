import tiktoken

class TikTokenWrapper:
    def __init__(self, model_name="gpt2"):
        """
        Wrapper around tiktoken tokenizer.
        Keeps tokenizer usage isolated from rest of code.
        """
        self.encoding = tiktoken.get_encoding(model_name)
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text):
        return self.encoding.encode(text)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

