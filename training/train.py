from tokenizer.tokenizer import TikTokenWrapper
from data.dataset import TextDataset

def main():
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = TikTokenWrapper(model_name="gpt2")
    print("Vocab size:", tokenizer.vocab_size)

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=32
    )

    x, y = dataset[0]

    print("Input shape:", x.shape)
    print("Decoded input:", tokenizer.decode(x.tolist()))

if __name__ == "__main__":
    main()

