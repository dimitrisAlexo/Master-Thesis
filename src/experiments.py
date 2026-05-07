"""
experiments.py — Run multiple LOSO experiments (10 repetitions) for any modality/model.

Usage:
    python experiments.py --type tremor --model baseline
    python experiments.py --type tremor --model simclr
    python experiments.py --type fmi    --model baseline
    python experiments.py --type fmi    --model simclr
    python experiments.py --type fusion --model baseline
    python experiments.py --type fusion --model simclr
"""

import argparse
import os
import gc
import json
import sys
import subprocess
import numpy as np
import pickle as pkl

import tremorSimCLRattentionMIL as tremor_mil
import typingSimCLRattentionMIL as typing_mil
import fusion as fusion_mod


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_tremor_dataset():
    try:
        with open("datasets/sdataset.pickle", "rb") as f:
            print("Loading tremor sdataset...")
            sdataset = pkl.load(f)
        print(f"windows shape: {sdataset['X'][0].shape}")
        return sdataset
    except FileNotFoundError:
        print("datasets/sdataset.pickle not found. Run dataset creation first.")
        return None


def load_typing_dataset():
    try:
        with open("datasets/typing_sdataset.pickle", "rb") as f:
            print("Loading typing sdataset...")
            sdataset = pkl.load(f)
        print(f"windows shape: {sdataset['X'][0].shape}")
        return sdataset
    except FileNotFoundError:
        print("datasets/typing_sdataset.pickle not found. Run dataset creation first.")
        return None


def load_additional_typing_dataset():
    try:
        with open("../data/additional_typing_sdataset.pickle", "rb") as f:
            print("Loading additional typing sdataset...")
            additional_sdataset = pkl.load(f)
        print(f"Additional dataset: {len(additional_sdataset)} subjects")
        return additional_sdataset
    except FileNotFoundError:
        print("additional_typing_sdataset.pickle not found. Training without additional data.")
        return None


def get_common_subject_ids():
    try:
        with open("datasets/fusion_dataset.pickle", "rb") as f:
            print("Loading fusion dataset to identify common subjects...")
            fusion_df = pkl.load(f)
        common_ids = set(fusion_df["subject_id"].tolist())
        print(f"Common subjects (tremor + typing): {len(common_ids)}")
        return common_ids
    except FileNotFoundError:
        print("datasets/fusion_dataset.pickle not found. Using all subjects for evaluation.")
        return None


# ── Multi-repetition runners ───────────────────────────────────────────────────

def run_multiple_tremor_loso_experiments(
    sdataset,
    repetitions=10,
    save_path="../results/tremor_results.json",
    restart_interval=1,
):
    """Run tremor loso_evaluate 'repetitions' times and save averaged metrics."""
    start_rep = int(os.environ.get("RESULTS_START_REP", "0"))

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            saved = json.load(f)
            accuracy_list     = saved.get("accuracy_list", [])
            sensitivity_list  = saved.get("sensitivity_list", [])
            specificity_list  = saved.get("specificity_list", [])
            precision_list    = saved.get("precision_list", [])
            f1_score_list     = saved.get("f1_score_list", [])
            start_iteration   = max(len(accuracy_list), start_rep)
    else:
        accuracy_list = sensitivity_list = specificity_list = []
        precision_list = f1_score_list = []
        accuracy_list, sensitivity_list, specificity_list, precision_list, f1_score_list = [], [], [], [], []
        start_iteration = start_rep

    def safe_mean_std(values):
        if len(values) == 0 or all(np.isnan(values)):
            return np.nan, np.nan
        return np.nanmean(values), np.nanstd(values)

    if start_iteration >= repetitions:
        print("All repetitions completed!")
        _print_final_stats(accuracy_list, sensitivity_list, specificity_list,
                           precision_list, f1_score_list)
        return

    end_iteration = min(start_iteration + restart_interval, repetitions)
    common_subject_ids = get_common_subject_ids()

    for i in range(start_iteration, end_iteration):
        print(f"\033[91mRepetition {i + 1}/{repetitions}\033[0m")
        try:
            _, _, _, results = tremor_mil.loso_evaluate(
                sdataset, eval_subject_ids=common_subject_ids
            )
            accuracy_list.append(results["final_accuracy"])
            sensitivity_list.append(results["final_sensitivity"])
            specificity_list.append(results["final_specificity"])
            precision_list.append(results["final_precision"])
            f1_score_list.append(results["final_f1_score"])
            with open(save_path, "w") as f:
                json.dump({
                    "accuracy_list":    accuracy_list,
                    "sensitivity_list": sensitivity_list,
                    "specificity_list": specificity_list,
                    "precision_list":   precision_list,
                    "f1_score_list":    f1_score_list,
                }, f)
        except Exception as e:
            print(f"Error during repetition {i + 1}: {e}")
        gc.collect()

    if end_iteration < repetitions:
        print(f"Completed {end_iteration} repetitions. Restarting process...")
        env = os.environ.copy()
        env["RESULTS_START_REP"] = str(end_iteration)
        subprocess.Popen([sys.executable] + sys.argv, env=env)
        sys.exit(0)

    _print_final_stats(accuracy_list, sensitivity_list, specificity_list,
                       precision_list, f1_score_list)


def run_multiple_typing_experiments(
    sdataset,
    repetitions=10,
    save_path="../results/fmi_results.json",
    restart_interval=1,
):
    """Run typing loso_evaluate 'repetitions' times and save averaged metrics."""
    start_rep = int(os.environ.get("RESULTS_START_REP", "0"))

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            saved = json.load(f)
            accuracy_list     = saved.get("accuracy_list", [])
            sensitivity_list  = saved.get("sensitivity_list", [])
            specificity_list  = saved.get("specificity_list", [])
            precision_list    = saved.get("precision_list", [])
            f1_score_list     = saved.get("f1_score_list", [])
            start_iteration   = max(len(accuracy_list), start_rep)
    else:
        accuracy_list, sensitivity_list, specificity_list, precision_list, f1_score_list = [], [], [], [], []
        start_iteration = start_rep

    if start_iteration >= repetitions:
        print("All repetitions completed!")
        _print_final_stats(accuracy_list, sensitivity_list, specificity_list,
                           precision_list, f1_score_list)
        return

    end_iteration = min(start_iteration + restart_interval, repetitions)
    additional_dataset = load_additional_typing_dataset()
    common_subject_ids = get_common_subject_ids()

    for i in range(start_iteration, end_iteration):
        print(f"\033[91mRepetition {i + 1}/{repetitions}\033[0m")
        try:
            _, _, _, results = typing_mil.loso_evaluate(
                sdataset,
                additional_data=additional_dataset,
                eval_subject_ids=common_subject_ids,
            )
            accuracy_list.append(results["final_accuracy"])
            sensitivity_list.append(results["final_sensitivity"])
            specificity_list.append(results["final_specificity"])
            precision_list.append(results["final_precision"])
            f1_score_list.append(results["final_f1_score"])
            with open(save_path, "w") as f:
                json.dump({
                    "accuracy_list":    accuracy_list,
                    "sensitivity_list": sensitivity_list,
                    "specificity_list": specificity_list,
                    "precision_list":   precision_list,
                    "f1_score_list":    f1_score_list,
                }, f)
        except Exception as e:
            print(f"Error during repetition {i + 1}: {e}")
        gc.collect()

    if end_iteration < repetitions:
        print(f"Completed {end_iteration} repetitions. Restarting process...")
        env = os.environ.copy()
        env["RESULTS_START_REP"] = str(end_iteration)
        subprocess.Popen([sys.executable] + sys.argv, env=env)
        sys.exit(0)

    _print_final_stats(accuracy_list, sensitivity_list, specificity_list,
                       precision_list, f1_score_list)


def _print_final_stats(accuracy_list, sensitivity_list, specificity_list,
                        precision_list, f1_score_list):
    def safe_mean_std(values):
        if len(values) == 0 or all(np.isnan(values)):
            return np.nan, np.nan
        return np.nanmean(values), np.nanstd(values)

    print("\nFINAL RESULTS:")
    for name, lst in [
        ("accuracy",    accuracy_list),
        ("sensitivity", sensitivity_list),
        ("specificity", specificity_list),
        ("precision",   precision_list),
        ("f1_score",    f1_score_list),
    ]:
        mean, std = safe_mean_std(lst)
        print(f"{name}_mean: {mean:.4f}, {name}_std: {std:.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 10-repetition LOSO experiments for tremor, FMI, or fusion."
    )
    parser.add_argument(
        "--type",
        choices=["tremor", "fmi", "fusion"],
        required=True,
        help="Experiment type",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "simclr"],
        required=True,
        help="Model mode: 'baseline' (no pretraining) or 'simclr' (with SimCLR pretraining)",
    )
    args = parser.parse_args()

    save_path = f"../results/{args.type}_{args.model}_results.json"

    # Setup environment (GPU, mixed precision)
    tremor_mil.setup_environment()

    if args.type == "tremor":
        tremor_mil.MODE = args.model
        sdataset = load_tremor_dataset()
        if sdataset is not None:
            run_multiple_tremor_loso_experiments(
                sdataset, repetitions=10, save_path=save_path
            )

    elif args.type == "fmi":
        typing_mil.MODE = args.model
        sdataset = load_typing_dataset()
        if sdataset is not None:
            run_multiple_typing_experiments(
                sdataset, repetitions=10, save_path=save_path
            )

    elif args.type == "fusion":
        fusion_mod.MODE = args.model
        fusion_mod.run_fusion_experiment(repetitions=10, save_path=save_path)
