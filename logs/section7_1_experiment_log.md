# Section 7.1 Experiment Log (Template)

Use this file as your official experiment record for Assignment 1 Section 7.

## 1. Logging Infrastructure

- Training script: `run_section7_training.py`
- Required per-run artifacts:
  - `runs/<experiment_name>_<timestamp>/config.json`
  - `runs/<experiment_name>_<timestamp>/metrics.csv`
  - `runs/<experiment_name>_<timestamp>/summary.json`
  - `runs/<experiment_name>_<timestamp>/checkpoints/checkpoint_step_*.pt`
- Required tracked metrics:
  - `step`
  - `wallclock_sec`
  - `train_loss`
  - `val_loss` (periodic)
  - `lr`

## 2. Reproducibility Metadata

- Date:
- Git commit:
- Device (GPU/CPU):
- Python / PyTorch version:
- Tokenizer files:
- Train token file:
- Val token file:
- Notes:

## 3. Baseline Run

| experiment_name | command | batch_size | max_iters | learning_rate | final_train_loss | final_val_loss | best_val_loss | best_step | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |

Curve files / references:

- Train-vs-step:
- Val-vs-step:
- Train-vs-wallclock:
- Val-vs-wallclock:

## 4. Section 7.2: Learning-Rate Sweep

| experiment_name | learning_rate | batch_size | max_iters | converged_or_diverged | final_val_loss | best_val_loss | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

Summary:

- Best learning rate:
- Evidence from curves:
- Divergence case and explanation:

## 5. Section 7.2: Batch-Size Study

| experiment_name | batch_size | learning_rate | max_iters | final_val_loss | best_val_loss | notes |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

Summary:

- Largest stable batch size on this hardware:
- Best speed/quality tradeoff:

## 6. Section 7.2: Text Generation Checks

| checkpoint | prompt | temperature | top_p | max_new_tokens | output_file_or_snippet | notes_on_fluency |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

## 7. Section 7.3: Ablations

| experiment_name | change_vs_baseline | command | final_val_loss | best_val_loss | notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

## 8. Final Conclusions

1. Which configuration performed best and why?
2. Which hyperparameter was most sensitive?
3. Which run failed/diverged and what did that teach you?
4. What would you try next with more compute?
