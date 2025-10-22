# Import functions but not execute main blocks
from typingSimCLRattentionMIL import (
    rkf_evaluate,
    loso_evaluate,
    setup_environment,
    MODE,
)
from tremorSimCLRattentionMIL import (
    rkf_evaluate as tremor_rkf_evaluate,
    loso_evaluate as tremor_loso_evaluate,
)
from utils import unpickle_data
import os
import gc
import numpy as np
import json
import pickle as pkl
import sys
import subprocess


def run_multiple_tremor_experiments(
    sdataset,
    k=5,
    n_repeats=5,
    repetitions=10,
    save_path="../results/results_tremor.json",
):
    """
    Run the rkf_evaluate experiment multiple times, calculate the average and standard deviation
    of accuracy, sensitivity, specificity, precision, and F1-score, ignoring NaN values, and
    save intermediate results to avoid data loss.
    """
    # Load existing results if available
    if os.path.exists(save_path):
        with open(save_path, "r") as file:
            saved_data = json.load(file)
            accuracy_list = saved_data.get("accuracy_list", [])
            sensitivity_list = saved_data.get("sensitivity_list", [])
            specificity_list = saved_data.get("specificity_list", [])
            precision_list = saved_data.get("precision_list", [])
            f1_score_list = saved_data.get("f1_score_list", [])
            start_iteration = len(accuracy_list)
    else:
        # Initialize lists to store metric values across repetitions
        accuracy_list = []
        sensitivity_list = []
        specificity_list = []
        precision_list = []
        f1_score_list = []
        start_iteration = 0

    # Helper function to safely compute mean and std, ignoring NaN values
    def safe_mean_std(values):
        if len(values) == 0 or all(np.isnan(values)):
            return np.nan, np.nan
        return np.nanmean(values), np.nanstd(values)

    # Run the experiment 'repetitions' times
    for i in range(start_iteration, repetitions):
        print(f"\033[91mRepetition {i + 1}/{repetitions}\033[0m")
        try:
            # Perform the rkf evaluation using tremor functions
            _, _, _, results = tremor_rkf_evaluate(sdataset, k=k, n_repeats=n_repeats)

            # Append results to lists
            accuracy_list.append(results["final_accuracy"])
            sensitivity_list.append(results["final_sensitivity"])
            specificity_list.append(results["final_specificity"])
            precision_list.append(results["final_precision"])
            f1_score_list.append(results["final_f1_score"])

            # Save results after every repetition
            with open(save_path, "w") as file:
                json.dump(
                    {
                        "accuracy_list": accuracy_list,
                        "sensitivity_list": sensitivity_list,
                        "specificity_list": specificity_list,
                        "precision_list": precision_list,
                        "f1_score_list": f1_score_list,
                    },
                    file,
                )

        except Exception as e:
            print(f"Error during repetition {i + 1}: {e}")

        # Memory management
        gc.collect()

    # Calculate mean and standard deviation for each metric
    accuracy_mean, accuracy_std = safe_mean_std(accuracy_list)
    sensitivity_mean, sensitivity_std = safe_mean_std(sensitivity_list)
    specificity_mean, specificity_std = safe_mean_std(specificity_list)
    precision_mean, precision_std = safe_mean_std(precision_list)
    f1_score_mean, f1_score_std = safe_mean_std(f1_score_list)

    # Create a metrics summary dictionary
    metrics_summary = {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "sensitivity_mean": sensitivity_mean,
        "sensitivity_std": sensitivity_std,
        "specificity_mean": specificity_mean,
        "specificity_std": specificity_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "f1_score_mean": f1_score_mean,
        "f1_score_std": f1_score_std,
    }

    # Print the metrics summary
    for metric, value in metrics_summary.items():
        if np.isnan(value):
            print(f"{metric}: No valid values (all NaN)")
        else:
            print(f"{metric}: {value:.4f}")

    return metrics_summary


def run_multiple_tremor_loso_experiments(
    sdataset,
    repetitions=10,
    save_path="../results/200_results_tremor_baseline.json",
    restart_interval=1,
):
    """
    Run the tremor loso_evaluate experiment multiple times, calculate the average and standard deviation
    of accuracy, sensitivity, specificity, precision, and F1-score, ignoring NaN values, and
    save intermediate results to avoid data loss. Restarts process every restart_interval repetitions.
    """
    # Check if we should restart the process
    start_rep = int(os.environ.get("RESULTS_START_REP", "0"))

    # Load existing results if available
    if os.path.exists(save_path):
        with open(save_path, "r") as file:
            saved_data = json.load(file)
            accuracy_list = saved_data.get("accuracy_list", [])
            sensitivity_list = saved_data.get("sensitivity_list", [])
            specificity_list = saved_data.get("specificity_list", [])
            precision_list = saved_data.get("precision_list", [])
            f1_score_list = saved_data.get("f1_score_list", [])
            start_iteration = max(len(accuracy_list), start_rep)
    else:
        # Initialize lists to store metric values across repetitions
        accuracy_list = []
        sensitivity_list = []
        specificity_list = []
        precision_list = []
        f1_score_list = []
        start_iteration = start_rep

    # If we've completed all repetitions, print final results and exit
    if start_iteration >= repetitions:
        print("All repetitions completed!")

        # Calculate and print final statistics
        def safe_mean_std(values):
            if len(values) == 0 or all(np.isnan(values)):
                return np.nan, np.nan
            return np.nanmean(values), np.nanstd(values)

        accuracy_mean, accuracy_std = safe_mean_std(accuracy_list)
        sensitivity_mean, sensitivity_std = safe_mean_std(sensitivity_list)
        specificity_mean, specificity_std = safe_mean_std(specificity_list)
        precision_mean, precision_std = safe_mean_std(precision_list)
        f1_score_mean, f1_score_std = safe_mean_std(f1_score_list)

        print("FINAL RESULTS:")
        print(f"accuracy_mean: {accuracy_mean:.4f}, accuracy_std: {accuracy_std:.4f}")
        print(
            f"sensitivity_mean: {sensitivity_mean:.4f}, sensitivity_std: {sensitivity_std:.4f}"
        )
        print(
            f"specificity_mean: {specificity_mean:.4f}, specificity_std: {specificity_std:.4f}"
        )
        print(
            f"precision_mean: {precision_mean:.4f}, precision_std: {precision_std:.4f}"
        )
        print(f"f1_score_mean: {f1_score_mean:.4f}, f1_score_std: {f1_score_std:.4f}")
        return

    # Calculate end iteration for this session
    end_iteration = min(start_iteration + restart_interval, repetitions)

    # Helper function to safely compute mean and std, ignoring NaN values
    def safe_mean_std(values):
        if len(values) == 0 or all(np.isnan(values)):
            return np.nan, np.nan
        return np.nanmean(values), np.nanstd(values)

    # Run the experiment for this session
    for i in range(start_iteration, end_iteration):
        print(f"\033[91mRepetition {i + 1}/{repetitions}\033[0m")
        try:
            # Perform the tremor loso evaluation
            _, _, _, results = tremor_loso_evaluate(sdataset)

            # Append results to lists
            accuracy_list.append(results["final_accuracy"])
            sensitivity_list.append(results["final_sensitivity"])
            specificity_list.append(results["final_specificity"])
            precision_list.append(results["final_precision"])
            f1_score_list.append(results["final_f1_score"])

            # Save results after every repetition
            with open(save_path, "w") as file:
                json.dump(
                    {
                        "accuracy_list": accuracy_list,
                        "sensitivity_list": sensitivity_list,
                        "specificity_list": specificity_list,
                        "precision_list": precision_list,
                        "f1_score_list": f1_score_list,
                    },
                    file,
                )

        except Exception as e:
            print(f"Error during repetition {i + 1}: {e}")

        # Memory management
        gc.collect()

    # If we haven't completed all repetitions, restart the process
    if end_iteration < repetitions:
        print(f"Completed {end_iteration} repetitions. Restarting process...")
        # Set environment variable for next start position
        env = os.environ.copy()
        env["RESULTS_START_REP"] = str(end_iteration)

        # Restart the script and exit current process immediately
        subprocess.Popen([sys.executable] + sys.argv, env=env)
        sys.exit(0)

    # Calculate mean and standard deviation for each metric (final run)
    accuracy_mean, accuracy_std = safe_mean_std(accuracy_list)
    sensitivity_mean, sensitivity_std = safe_mean_std(sensitivity_list)
    specificity_mean, specificity_std = safe_mean_std(specificity_list)
    precision_mean, precision_std = safe_mean_std(precision_list)
    f1_score_mean, f1_score_std = safe_mean_std(f1_score_list)

    # Create a metrics summary dictionary
    metrics_summary = {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "sensitivity_mean": sensitivity_mean,
        "sensitivity_std": sensitivity_std,
        "specificity_mean": specificity_mean,
        "specificity_std": specificity_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "f1_score_mean": f1_score_mean,
        "f1_score_std": f1_score_std,
    }

    # Print the metrics summary
    for metric, value in metrics_summary.items():
        if np.isnan(value):
            print(f"{metric}: No valid values (all NaN)")
        else:
            print(f"{metric}: {value:.4f}")

    return metrics_summary


def run_multiple_typing_experiments(
    sdataset,
    repetitions=10,
    save_path="../results/500_results_typing_pretrained.json",
    restart_interval=1,
):
    """
    Run the loso_evaluate experiment multiple times, calculate the average and standard deviation
    of accuracy, sensitivity, specificity, precision, and F1-score, ignoring NaN values, and
    save intermediate results to avoid data loss. Restarts process every restart_interval repetitions.
    """
    # Check if we should restart the process
    start_rep = int(os.environ.get("RESULTS_START_REP", "0"))

    # Load existing results if available
    if os.path.exists(save_path):
        with open(save_path, "r") as file:
            saved_data = json.load(file)
            accuracy_list = saved_data.get("accuracy_list", [])
            sensitivity_list = saved_data.get("sensitivity_list", [])
            specificity_list = saved_data.get("specificity_list", [])
            precision_list = saved_data.get("precision_list", [])
            f1_score_list = saved_data.get("f1_score_list", [])
            start_iteration = max(len(accuracy_list), start_rep)
    else:
        # Initialize lists to store metric values across repetitions
        accuracy_list = []
        sensitivity_list = []
        specificity_list = []
        precision_list = []
        f1_score_list = []
        start_iteration = start_rep

    # If we've completed all repetitions, print final results and exit
    if start_iteration >= repetitions:
        print("All repetitions completed!")

        # Calculate and print final statistics
        def safe_mean_std(values):
            if len(values) == 0 or all(np.isnan(values)):
                return np.nan, np.nan
            return np.nanmean(values), np.nanstd(values)

        accuracy_mean, accuracy_std = safe_mean_std(accuracy_list)
        sensitivity_mean, sensitivity_std = safe_mean_std(sensitivity_list)
        specificity_mean, specificity_std = safe_mean_std(specificity_list)
        precision_mean, precision_std = safe_mean_std(precision_list)
        f1_score_mean, f1_score_std = safe_mean_std(f1_score_list)

        print("FINAL RESULTS:")
        print(f"accuracy_mean: {accuracy_mean:.4f}, accuracy_std: {accuracy_std:.4f}")
        print(
            f"sensitivity_mean: {sensitivity_mean:.4f}, sensitivity_std: {sensitivity_std:.4f}"
        )
        print(
            f"specificity_mean: {specificity_mean:.4f}, specificity_std: {specificity_std:.4f}"
        )
        print(
            f"precision_mean: {precision_mean:.4f}, precision_std: {precision_std:.4f}"
        )
        print(f"f1_score_mean: {f1_score_mean:.4f}, f1_score_std: {f1_score_std:.4f}")
        return

    # Calculate end iteration for this session
    end_iteration = min(start_iteration + restart_interval, repetitions)

    # Helper function to safely compute mean and std, ignoring NaN values
    def safe_mean_std(values):
        if len(values) == 0 or all(np.isnan(values)):
            return np.nan, np.nan
        return np.nanmean(values), np.nanstd(values)

    # Run the experiment for this session
    for i in range(start_iteration, end_iteration):
        print(f"\033[91mRepetition {i + 1}/{repetitions}\033[0m")
        try:
            # Perform the loso evaluation
            _, _, _, results = loso_evaluate(sdataset)

            # Append results to lists
            accuracy_list.append(results["final_accuracy"])
            sensitivity_list.append(results["final_sensitivity"])
            specificity_list.append(results["final_specificity"])
            precision_list.append(results["final_precision"])
            f1_score_list.append(results["final_f1_score"])

            # Save results after every repetition
            with open(save_path, "w") as file:
                json.dump(
                    {
                        "accuracy_list": accuracy_list,
                        "sensitivity_list": sensitivity_list,
                        "specificity_list": specificity_list,
                        "precision_list": precision_list,
                        "f1_score_list": f1_score_list,
                    },
                    file,
                )

        except Exception as e:
            print(f"Error during repetition {i + 1}: {e}")

        # Memory management
        gc.collect()

    # If we haven't completed all repetitions, restart the process
    if end_iteration < repetitions:
        print(f"Completed {end_iteration} repetitions. Restarting process...")
        # Set environment variable for next start position
        env = os.environ.copy()
        env["RESULTS_START_REP"] = str(end_iteration)

        # Restart the script and exit current process immediately
        subprocess.Popen([sys.executable] + sys.argv, env=env)
        sys.exit(0)

    # Calculate mean and standard deviation for each metric (final run)
    accuracy_mean, accuracy_std = safe_mean_std(accuracy_list)
    sensitivity_mean, sensitivity_std = safe_mean_std(sensitivity_list)
    specificity_mean, specificity_std = safe_mean_std(specificity_list)
    precision_mean, precision_std = safe_mean_std(precision_list)
    f1_score_mean, f1_score_std = safe_mean_std(f1_score_list)

    # Create a metrics summary dictionary
    metrics_summary = {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "sensitivity_mean": sensitivity_mean,
        "sensitivity_std": sensitivity_std,
        "specificity_mean": specificity_mean,
        "specificity_std": specificity_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "f1_score_mean": f1_score_mean,
        "f1_score_std": f1_score_std,
    }

    # Print the metrics summary
    for metric, value in metrics_summary.items():
        if np.isnan(value):
            print(f"{metric}: No valid values (all NaN)")
        else:
            print(f"{metric}: {value:.4f}")

    return metrics_summary


def load_typing_dataset():
    """Load the typing supervised dataset"""
    try:
        with open("typing_sdataset.pickle", "rb") as f:
            print("Loading typing sdataset...")
            sdataset = pkl.load(f)
        print(f"windows shape: {sdataset['X'][0].shape}")
        return sdataset
    except FileNotFoundError:
        print(
            "typing_sdataset.pickle not found. Please run the dataset creation first."
        )
        return None


def load_tremor_dataset():
    """Load the tremor supervised dataset"""
    try:
        with open("sdataset.pickle", "rb") as f:
            print("Loading tremor sdataset...")
            sdataset = pkl.load(f)
            print(f"windows shape: {sdataset['X'][0].shape}")
        return sdataset
    except FileNotFoundError:
        print("sdataset.pickle not found. Please run the dataset creation first.")
        return None


if __name__ == "__main__":
    # Setup environment
    start = setup_environment()

    print("RESULTS")

    # Load datasets
    typing_sdataset = load_typing_dataset()
    # tremor_sdataset = load_tremor_dataset()

    # Run typing experiments if dataset is available
    if typing_sdataset is not None:
        print("Running typing experiments...")
        run_multiple_typing_experiments(typing_sdataset, repetitions=10)

    # Run tremor experiments if dataset is available (uncomment to run)
    # if tremor_sdataset is not None:
    #     # print("Running tremor RKF experiments...")
    #     # run_multiple_tremor_experiments(tremor_sdataset, k=5, n_repeats=5, repetitions=10)

    #     print("Running tremor LOSO experiments...")
    #     run_multiple_tremor_loso_experiments(tremor_sdataset, repetitions=10)
