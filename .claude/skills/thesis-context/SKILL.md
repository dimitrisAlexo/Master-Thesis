---
name: thesis-context
description: Context for the Master-Thesis repo (Parkinson's detection by fusing tremor IMU + typing dynamics with SimCLR pretraining). Load at the start of any work on this repo — fusion.py, the tremor/typing branches, subject-level SimCLR, LOSO experiments or results.
---

# Master-Thesis — project context

## Goal
Detect Parkinson's-related signs per subject with multiple-instance learning (MIL) on two modalities, then fuse them:
- **Tremor**: wrist accelerometer windows (bag = 200 windows × 1000 × 3) → label tremor.
- **Typing**: keystroke hold/flight-time histograms (bag = 500 × 502) → label FMI (fine-motor impairment).
- **Fusion**: both bags → multi-label [Tremor, FMI, PD]. PD == FMI on all 22 fusion subjects.

Self-supervised pretraining (SimCLR) on unlabeled data, then LOSO evaluation. The current research question is **subject-level SimCLR** (`MODE="subject_simclr"`); the published paper used window-level SimCLR (`MODE="simclr"`): SSL fused tremor F1 0.944 / FMI 0.919 / PD 0.926. Aim: make subject_simclr fusion ≥ its single branches (standalone tremor ≈0.91, FMI ≈0.90).

## Repo map (`src/`, run everything from `src/`)
- `tremorSubjectSimCLR.py`, `typingSubjectSimCLR.py` — subject-level contrastive pretraining (MoCo queue); write `weights/{tremor,typing}/*_subject_simclr_*`.
- `tremorSimCLRattentionMIL.py`, `typingSimCLRattentionMIL.py` — MIL models + `train()` recipes per mode + standalone LOSO.
- `fusion.py` — per LOSO fold: trains both branches (`run_pretraining`), then `FusionModel` = frozen branches → L2-norm → concat → MLP head (3 sigmoid outputs).
- `experiments.py` — `python experiments.py --type {tremor,fmi,fusion} --model {baseline,simclr,subject_simclr}`; repeated runs, resumes from `../results/<type>_<model>_results.json` (move that file before re-running a changed config).
- Data: `src/datasets/*.pickle`, `../data/`. Labeled: 64 tremor, 25 typing (+33 additional), 22 subjects with both.

## Facts that matter
- Fusion cohort (22): 11 tremor+ (all also FMI+), 8 healthy, **3 FMI-only = subjects 1, 4, 13** (1-indexed fold numbers). The fusion head learns "FMI ⇒ tremor" → tremor false positives on 4/13. Often-lost subjects: 1, 4, 5, 10, 13, 15, 19, 21.
- One subject = ~0.045 F1; single-run F1 swings ±0.05. Compare configs by the mean over ≥3 full runs.
- GPU is a 4 GB RTX 3050: one full fusion LOSO run ≈ 40 min; datasets must be built on CPU (`tf.device("/CPU:0")`) and `TF_FORCE_GPU_ALLOW_GROWTH=true`, otherwise OOM.
- Code comments are not always correct — verify against code/data.

## Current code state (Oct 2026)
- Padding-mask fix in `MILModel.create_mask_layer` for subject_simclr only (normalized padding was never masked).
- Fusion phase-2 fine-tune off by default (`FUSION_FT_EPOCHS` env to enable); fusion head = concat, loss 3·tremor + fmi + pd.
- Tremor branch in fusion: batch 4, LP-FT (phase 2: 20 ep @ 1e-4).

## Experiment history (don't redo blindly)
Full report: `results/claude_fusion_experiments.json` (18 configs, 58 full LOSO runs); infrastructure in `~/code/claude_experiments/`.
- Tried without robust gain: add vs concat, loss weights, head epochs/LR/dropout/batch, label balancing, cross-subject pairing, leak-free branches, typing pretraining augmentation (better probe, same downstream), alternative tremor pretraining (no `permute_segments`, lr/queue changes — seed variance as large as the effect). Linear-probe AUC does not predict downstream results.
- Stronger tremor fine-tune (batch 8, 50 ep @ 1e-3) + typing embedding ×0.3 looked best in 3-run tests (tremor ~0.94, FMI ~0.86) but the user's full run was **worse** → reverted; treat as unconfirmed.
- FMI is capped (~0.84–0.92) by typing-branch errors on subjects 1, 5, 15, 21; subject 19 is a tremor miss in almost every setup.
