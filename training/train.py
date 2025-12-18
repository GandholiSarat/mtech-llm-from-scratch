from data.dataset import TextDataset, CharTokenizer

def main():
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=32
    )

    x, y = dataset[0]

    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    print("Decoded input:", tokenizer.decode(x.tolist()))
    print("Decoded target:", tokenizer.decode(y.tolist()))

if __name__ == "__main__":
    main()

