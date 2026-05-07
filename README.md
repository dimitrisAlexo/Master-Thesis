# Thesis
In-The-Wild Detection of Intermittent Parkinsonian Tremor: A Federated, Self-Supervised Learning Approach Using Attention-Based MIL

> Download the datasets from https://zenodo.org/records/7273759 and https://zenodo.org/records/4311175

## Running Experiments

All scripts are run from the `src/` directory:

```bash
cd src/
```

### Multi-LOSO (10 repetitions) — `experiments.py`

Runs the full evaluation loop, saves averaged metrics to `results/`.

| Command | Description |
|---|---|
| `python experiments.py --type tremor --model baseline` | Baseline tremor classifier (no pretraining) |
| `python experiments.py --type tremor --model simclr` | SSL tremor classifier (SimCLR pretraining) |
| `python experiments.py --type fmi --model baseline` | Baseline FMI classifier (no pretraining) |
| `python experiments.py --type fmi --model simclr` | SSL FMI classifier (SimCLR pretraining) |
| `python experiments.py --type fusion --model baseline` | Baseline fused classifier |
| `python experiments.py --type fusion --model simclr` | SSL fused classifier |

Results are saved to `results/<type>_<model>_results.json`.

### Single LOSO run — individual scripts

Runs one LOSO evaluation pass (useful for debugging or quick checks).

| Command | Description |
|---|---|
| `python tremorSimCLRattentionMIL.py --model baseline` | Single tremor LOSO, no pretraining |
| `python tremorSimCLRattentionMIL.py --model simclr` | Single tremor LOSO, SimCLR pretraining |
| `python typingSimCLRattentionMIL.py --model baseline` | Single FMI LOSO, no pretraining |
| `python typingSimCLRattentionMIL.py --model simclr` | Single FMI LOSO, SimCLR pretraining |
| `python fusion.py --model baseline` | Single fusion LOSO, no pretraining |
| `python fusion.py --model simclr` | Single fusion LOSO, SimCLR pretraining |

`fusion.py` also accepts `--bimodal` (use bimodal SimCLR weights for typing branch) and `--debug` (run only 6 critical folds).

### Pretraining (required before SSL runs)

SimCLR pretraining must be run before SSL experiments if weights are not already present:

| Command | Output weights |
|---|---|
| `python tremorSimCLR.py` | `weights/tremor/tremor_simclr_embeddings.weights.h5` |
| `python typingSimCLR.py` | `weights/typing/typing_simclr_embeddings.weights.h5` |
| `python bimodalSimCLR.py` | `weights/fusion/typing_bimodal_embeddings.weights.h5` |
