"""
predict.py - Interactive Text Generation for DyadicPush Language Model
======================================================================
Loads a trained DyadicPush model checkpoint and generates text autoregressively.
Supports:
- Command-line prompt generation: python predict.py --prompt "Once upon a time"
- Interactive terminal chat / prompt mode when run without --prompt
- Temperature, top-k, and nucleus (top-p) sampling
- Zero KV-cache footprint inference
"""

import argparse
import os
import sys
import torch
from model import DyadicPushConfig, DyadicPushLM
from tokenizer import BPETokenizerWrapper, CharTokenizer


def load_model_and_tokenizer(checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' does not exist. "
            "Please train the model first with 'python train.py'."
        )

    print(f"[*] Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 1. Rebuild tokenizer
    tok_info = checkpoint.get("tokenizer", {})
    tok_type = tok_info.get("type", "unknown")

    if tok_type == "bpe":
        tokenizer = BPETokenizerWrapper.from_json_str(tok_info["data"])
    elif tok_type == "char":
        tokenizer = CharTokenizer.from_dict(tok_info)
    else:
        # Fallback: check if local tokenizer.json exists
        tok_json_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.json")
        if os.path.exists(tok_json_path):
            tokenizer = BPETokenizerWrapper.from_file(tok_json_path)
        else:
            raise RuntimeError("Could not reconstruct tokenizer from checkpoint.")

    # 2. Rebuild model
    config_dict = checkpoint["config"]
    config = DyadicPushConfig(**config_dict)
    model = DyadicPushLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    num_params = model.get_num_params()
    val_loss = checkpoint.get("val_loss", None)
    step = checkpoint.get("step", None)
    print(f"[*] Model loaded: {num_params:,} parameters (~{num_params / 1e6:.2f}M)")
    if step is not None:
        print(f"[*] Trained for {step} steps" + (f" (best val loss: {val_loss:.4f})" if val_loss else ""))

    return model, tokenizer, config


def generate_streaming(
    model: DyadicPushLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
):
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [0]
    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    print("-" * 60)
    print(prompt, end="", flush=True)

    generated_indices = list(encoded)
    for _ in range(max_new_tokens):
        # Crop context window
        idx_cond = idx if idx.size(1) <= model.config.max_seq_len else idx[:, -model.config.max_seq_len:]
        with torch.no_grad():
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_id = next_token.item()

        # Decode token and stream to stdout
        token_str = tokenizer.decode([next_id])
        print(token_str, end="", flush=True)

        idx = torch.cat((idx, next_token), dim=1)
        generated_indices.append(next_id)

    print("\n" + "-" * 60)
    return tokenizer.decode(generated_indices)


def main():
    parser = argparse.ArgumentParser(description="Predict / Generate text using DyadicPush LM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text for single prediction")
    parser.add_argument("--max_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering threshold")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) cumulative probability")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.device)

    if args.prompt is not None:
        generate_streaming(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )
    else:
        print("\n" + "=" * 60)
        print("DyadicPush LM: Interactive Prediction Mode")
        print("Enter a prompt and press Enter to generate text.")
        print("Type 'exit' or press Ctrl+C to terminate.")
        print("=" * 60)

        while True:
            try:
                user_prompt = input("\nPrompt > ").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("exit", "quit", "q"):
                    print("Exiting.")
                    break
                generate_streaming(
                    model,
                    tokenizer,
                    prompt=user_prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device,
                )
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
