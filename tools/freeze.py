def freeze_gpt2_except_ln_and_head(model):
    """
    Freeze all GPT-2 parameters except:
    - LayerNorms
    - lm_head
    """

    for name, param in model.named_parameters():
        param.requires_grad = False

        if (
            "ln_" in name
            or "ln1" in name
            or "ln2" in name
            or "ln_f" in name
            or "lm_head" in name
        ):
            param.requires_grad = True

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())

    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

