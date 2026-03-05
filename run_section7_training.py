#!/usr/bin/env python3
"""Section 7 training script with experiment logging.

This script adds run tracking that is expected by Assignment 1 Section 7.1:
- periodic validation loss evaluation
- logging by gradient steps and wallclock time
- reproducible run configs
- checkpoint saving
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cs336_basics.MyAdamW import MyAdamW
from cs336_basics.MyTransformerLM import MyTransformerLM
from cs336_basics.utils import (
    cross_entropy,
    get_batch,
    gradient_clipping,
    load_checkpoint,
    lr_cosine_schedule,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM with Section 7.1 logging.")

    # Run metadata
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="runs")
    parser.add_argument("--notes", type=str, default="")

    # Data
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--data_dtype", type=str, default="uint16", choices=["uint16", "int32", "int64"])

    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_learning_rate", type=float, default=None)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Logging / eval / checkpoints
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            print("MPS not available, falling back to CPU.")
            return torch.device("cpu")
    return torch.device(requested)


def load_token_array(path: str, data_dtype: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.load(path, mmap_mode="r")
    elif ext == ".bin":
        dtype_map = {
            "uint16": np.uint16,
            "int32": np.int32,
            "int64": np.int64,
        }
        data = np.memmap(path, dtype=dtype_map[data_dtype], mode="r")
    else:
        raise ValueError("Data path must end with .npy or .bin")

    if data.ndim != 1:
        raise ValueError(f"Data must be 1D token IDs. Got shape={data.shape}.")
    return data


def init_matrix(shape: tuple[int, ...]) -> Tensor:
    t = torch.empty(shape, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(t)
    return t


def build_model(args: argparse.Namespace, device: torch.device) -> MyTransformerLM:
    d_ff = 4 * args.d_model
    init_weights: dict[str, Tensor] = {
        "token_embeddings.weight": init_matrix((args.vocab_size, args.d_model)),
        "ln_final.weight": torch.ones(args.d_model, dtype=torch.float32),
        "lm_head.weight": init_matrix((args.vocab_size, args.d_model)),
    }

    for layer_idx in range(args.num_layers):
        prefix = f"layers.{layer_idx}."
        init_weights[prefix + "attn.q_proj.weight"] = init_matrix((args.d_model, args.d_model))
        init_weights[prefix + "attn.k_proj.weight"] = init_matrix((args.d_model, args.d_model))
        init_weights[prefix + "attn.v_proj.weight"] = init_matrix((args.d_model, args.d_model))
        init_weights[prefix + "attn.output_proj.weight"] = init_matrix((args.d_model, args.d_model))
        init_weights[prefix + "ln1.weight"] = torch.ones(args.d_model, dtype=torch.float32)
        init_weights[prefix + "ln2.weight"] = torch.ones(args.d_model, dtype=torch.float32)
        init_weights[prefix + "ffn.w1.weight"] = init_matrix((d_ff, args.d_model))
        init_weights[prefix + "ffn.w2.weight"] = init_matrix((args.d_model, d_ff))
        init_weights[prefix + "ffn.w3.weight"] = init_matrix((d_ff, args.d_model))

    model = MyTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        weights=init_weights,
    )
    return model.to(device)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def evaluate(
    model: torch.nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = get_batch(
                data=data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(x)
            loss = cross_entropy(logits, y)
            losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / len(losses))


def as_jsonable_config(args: argparse.Namespace, resolved_device: torch.device, min_lr: float) -> dict[str, Any]:
    cfg = vars(args).copy()
    cfg["resolved_device"] = str(resolved_device)
    cfg["min_learning_rate_effective"] = min_lr
    cfg["command"] = " ".join(sys.argv)
    cfg["started_at_utc"] = datetime.utcnow().isoformat(timespec="seconds")
    return cfg


def main() -> None:
    args = parse_args()

    if args.num_heads <= 0 or args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if args.max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if args.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    if args.eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")
    if args.save_interval <= 0:
        raise ValueError("save_interval must be positive.")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = load_token_array(args.train_data_path, args.data_dtype)
    val_data = load_token_array(args.val_data_path, args.data_dtype)
    if len(train_data) <= args.context_length:
        raise ValueError("Training data is too short for the configured context_length.")
    if len(val_data) <= args.context_length:
        raise ValueError("Validation data is too short for the configured context_length.")

    model = build_model(args, device)
    optimizer = MyAdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    min_lr = args.min_learning_rate if args.min_learning_rate is not None else args.learning_rate * 0.1

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer) + 1
        move_optimizer_state_to_device(optimizer, device)
        print(f"Resumed from {args.resume_from}; starting at step {start_iter}.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / f"{args.experiment_name}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = as_jsonable_config(args, device, float(min_lr))
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    metrics_path = run_dir / "metrics.csv"
    start_wall = time.perf_counter()
    best_val_loss = float("inf")
    best_step = -1
    model.train()

    with metrics_path.open("w", newline="", encoding="utf-8") as f_metrics:
        writer = csv.DictWriter(
            f_metrics,
            fieldnames=["step", "wallclock_sec", "train_loss", "val_loss", "lr"],
        )
        writer.writeheader()

        for step in range(start_iter, args.max_iters):
            lr = lr_cosine_schedule(
                t=step,
                max_lr=args.learning_rate,
                min_lr=min_lr,
                t_warm_up=args.warmup_iters,
                t_c=args.max_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(
                data=train_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
            )
            logits = model(x)
            loss = cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()

            step_num = step + 1
            do_eval = (step_num % args.eval_interval == 0) or (step == start_iter) or (step_num == args.max_iters)
            val_loss = None
            if do_eval:
                val_loss = evaluate(
                    model=model,
                    data=val_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=device,
                    eval_batches=args.eval_batches,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step_num

            should_log = (step_num % args.log_interval == 0) or do_eval
            if should_log:
                wall = time.perf_counter() - start_wall
                writer.writerow(
                    {
                        "step": step_num,
                        "wallclock_sec": f"{wall:.6f}",
                        "train_loss": f"{loss.item():.8f}",
                        "val_loss": "" if val_loss is None else f"{val_loss:.8f}",
                        "lr": f"{lr:.10f}",
                    }
                )
                f_metrics.flush()

                if val_loss is None:
                    print(f"step {step_num}/{args.max_iters} | train_loss {loss.item():.6f} | lr {lr:.8f}")
                else:
                    print(
                        f"step {step_num}/{args.max_iters} | "
                        f"train_loss {loss.item():.6f} | val_loss {val_loss:.6f} | lr {lr:.8f}"
                    )

            if (step_num % args.save_interval == 0) or (step_num == args.max_iters):
                ckpt_path = ckpt_dir / f"checkpoint_step_{step_num}.pt"
                save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=str(ckpt_path))
                print(f"Saved checkpoint to {ckpt_path}")

    total_wall = time.perf_counter() - start_wall
    summary = {
        "run_dir": str(run_dir),
        "total_wallclock_sec": total_wall,
        "best_val_loss": None if best_step < 0 else best_val_loss,
        "best_step": None if best_step < 0 else best_step,
        "finished_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Run complete. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
