import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    x = torch.randn(2, 3)
    y = torch.randn(3, 4)

    z = x @ y
    print("Matrix multiply works:", z.shape)

if __name__ == "__main__":
    main()

