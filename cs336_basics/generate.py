import torch
import torch.nn.functional as F


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """
    Generate text from a language model using temperature + top-p sampling.

    Args:
        model: Trained autoregressive language model.
        tokenizer: Tokenizer exposing encode(text)->list[int] and decode(ids)->str.
        prompt: Prompt text.
        max_new_tokens: Number of tokens to generate at most.
        temperature: Logit temperature (> 0).
        top_p: Nucleus sampling threshold (0 < top_p <= 1).
        device: Inference device string.

    Returns:
        The decoded prompt + generated continuation.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1].")

    model = model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt encodes to an empty token sequence.")

    # Prefer tokenizer's registered special-token map when available.
    eos_token_id = None
    if hasattr(tokenizer, "special_tokens") and isinstance(tokenizer.special_tokens, dict):
        eos_token_id = tokenizer.special_tokens.get("<|endoftext|>")
    if eos_token_id is None:
        eos_ids = tokenizer.encode("<|endoftext|>")
        if len(eos_ids) == 1:
            eos_token_id = eos_ids[0]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    context_length = getattr(model, "context_length", None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Keep generation window within model context.
            model_input = input_ids
            if isinstance(context_length, int) and context_length > 0 and model_input.size(1) > context_length:
                model_input = model_input[:, -context_length:]

            logits = model(model_input)[:, -1, :]  # (1, vocab_size)
            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            # Top-p (nucleus) filtering.
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)

            probs_sum = sorted_probs.sum(dim=-1, keepdim=True)
            if torch.any(probs_sum == 0):
                # Numerical safety fallback: keep highest-prob token.
                sorted_probs[..., 0] = 1.0
                probs_sum = sorted_probs.sum(dim=-1, keepdim=True)
            sorted_probs = sorted_probs / probs_sum

            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)  # index in sorted space
            next_token = sorted_indices.gather(-1, sampled_idx)  # map back to vocab index

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return tokenizer.decode(input_ids[0].tolist())
