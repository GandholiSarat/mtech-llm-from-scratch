import torch
from transformers import GPT2LMHeadModel

from model.gpt2_compatible import build_gpt2_compatible_model


def load_gpt2_weights():
    """
    Load GPT-2 pretrained weights from HuggingFace
    into our custom GPTModel implementation.
    """

    # -------------------------------------------------
    # Load HuggingFace GPT-2 (weights only)
    # -------------------------------------------------
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_state = hf_model.state_dict()

    # -------------------------------------------------
    # Build our GPT-2 compatible model
    # -------------------------------------------------
    model = build_gpt2_compatible_model()
    our_state = model.state_dict()

    mapped_state = {}

    for name in our_state:

        # -------------------------------------------------
        # Skip buffers (not trainable parameters)
        # -------------------------------------------------
        if "causal_mask" in name:
            continue

        # -------------------------------------------------
        # Token embeddings
        # -------------------------------------------------
        if name == "embeddings.token_embedding.weight":
            mapped_state[name] = hf_state["transformer.wte.weight"]
            continue

        # -------------------------------------------------
        # Positional embeddings (add batch dim)
        # -------------------------------------------------
        if name == "embeddings.position_embedding":
            mapped_state[name] = hf_state[
                "transformer.wpe.weight"
            ].unsqueeze(0)
            continue

        # -------------------------------------------------
        # Final layer norm
        # -------------------------------------------------
        if name.startswith("ln_f."):
            hf_name = name.replace("ln_f.", "transformer.ln_f.")
            mapped_state[name] = hf_state[hf_name]
            continue

        # -------------------------------------------------
        # Transformer blocks
        # -------------------------------------------------
        if name.startswith("blocks."):

            hf_name = name
            hf_name = hf_name.replace("blocks.", "transformer.h.")
            hf_name = hf_name.replace(".ln1.", ".ln_1.")
            hf_name = hf_name.replace(".ln2.", ".ln_2.")

            # -------------------------
            # LayerNorms
            # -------------------------
            if ".ln1.weight" in name or ".ln1.bias" in name:
                mapped_state[name] = hf_state[hf_name]
                continue

            if ".ln2.weight" in name or ".ln2.bias" in name:
                mapped_state[name] = hf_state[hf_name]
                continue

            # -------------------------
            # QKV projection
            # -------------------------
            if "attn.qkv_proj.weight" in name:
                hf_name = hf_name.replace(
                    "attn.qkv_proj.weight",
                    "attn.c_attn.weight"
                )
                mapped_state[name] = hf_state[hf_name].t()
                continue

            if "attn.qkv_proj.bias" in name:
                hf_name = hf_name.replace(
                    "attn.qkv_proj.bias",
                    "attn.c_attn.bias"
                )
                mapped_state[name] = hf_state[hf_name]
                continue

            # -------------------------
            # Attention output projection
            # -------------------------
            if "attn.out_proj.weight" in name:
                hf_name = hf_name.replace(
                    "attn.out_proj.weight",
                    "attn.c_proj.weight"
                )
                # HF GPT-2 uses Conv1D weights shaped (in_features, out_features).
                # nn.Linear expects (out_features, in_features).
                mapped_state[name] = hf_state[hf_name].t()
                continue

            if "attn.out_proj.bias" in name:
                hf_name = hf_name.replace(
                    "attn.out_proj.bias",
                    "attn.c_proj.bias"
                )
                mapped_state[name] = hf_state[hf_name]
                continue

            # -------------------------
            # Feedforward / MLP
            # -------------------------
            if "ffn.net.0.weight" in name:
                hf_name = hf_name.replace(
                    "ffn.net.0.weight",
                    "mlp.c_fc.weight"
                )
                mapped_state[name] = hf_state[hf_name].t()
                continue

            if "ffn.net.0.bias" in name:
                hf_name = hf_name.replace(
                    "ffn.net.0.bias",
                    "mlp.c_fc.bias"
                )
                mapped_state[name] = hf_state[hf_name]
                continue

            if "ffn.net.2.weight" in name:
                hf_name = hf_name.replace(
                    "ffn.net.2.weight",
                    "mlp.c_proj.weight"
                )
                mapped_state[name] = hf_state[hf_name].t()
                continue

            if "ffn.net.2.bias" in name:
                hf_name = hf_name.replace(
                    "ffn.net.2.bias",
                    "mlp.c_proj.bias"
                )
                mapped_state[name] = hf_state[hf_name]
                continue

        # -------------------------------------------------
        # Language model head
        # -------------------------------------------------
        if name == "lm_head.weight":
            mapped_state[name] = hf_state["lm_head.weight"]
            continue

        # -------------------------------------------------
        # Anything else
        # -------------------------------------------------
        print("Skipping:", name)

    # -------------------------------------------------
    # Load mapped weights
    # -------------------------------------------------
    # IMPORTANT: tie embeddings like GPT-2
    model.lm_head.weight = model.embeddings.token_embedding.weight
    model.load_state_dict(mapped_state, strict=False)
    # Disable dropout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    model.eval()

    return model

