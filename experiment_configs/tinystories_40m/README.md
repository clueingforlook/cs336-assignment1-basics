# TinyStories 40M Experiment Configs

These configs assume the 40.96M-token budget:

- `batch_size * max_iters * context_length = 40,960,000`
- `context_length = 256`

Run from the repository root:

```bash
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/baseline_40m.json
```

Learning-rate sweep:

```bash
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/lr_2e-4.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/lr_3e-4.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/lr_5e-4.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/lr_7e-4.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/lr_1e-3.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/lr_2e-3.json
```

Batch-size sweep:

```bash
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/batch_16.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/batch_32.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/batch_64.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/batch_128.json
```

Ablations:

```bash
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/ablate_no_norm_best_lr.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/ablate_no_norm_low_lr.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/ablate_post_norm.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/ablate_nope.json
./run_with_json.sh --script ./run_section7_training.py --config ./experiment_configs/tinystories_40m/ablate_silu_param_matched.json
```

Notes:

- The default training script behavior is preserved: `d_ff = 4 * d_model = 2048`.
- `ablate_silu_param_matched.json` uses `d_ff = 3072` so that SiLU has roughly the same FFN parameter count as the current `SwiGLU + d_ff=2048` baseline.
- If you want exact handout-style SwiGLU/SiLU parameter matching, switch to `SwiGLU d_ff=1344` vs `SiLU d_ff=2048`.
