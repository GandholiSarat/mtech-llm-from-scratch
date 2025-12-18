import torch
from torch.utils.data import Dataset

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return "".join([self.itos[t] for t in tokens])


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, context_length):
        """
        text: full input text (string)
        tokenizer: object with encode() method
        context_length: number of tokens in input
        """
        self.tokens = tokenizer.encode(text)
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_length]
        y = self.tokens[idx + 1 : idx + self.context_length + 1]

        return torch.tensor(x), torch.tensor(y)

