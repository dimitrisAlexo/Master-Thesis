# from simCLRattentionMIL import *
from typingSimCLRattentionMIL import *

import json


def run_multiple_tremor_experiments(
    sdataset, k=5, n_repeats=5, repetitions=10, save_path="results.json"
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
            # Perform the rkf evaluation
            _, _, _, results = rkf_evaluate(sdataset, k=k, n_repeats=n_repeats)

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


def run_multiple_typing_experiments(
    sdataset, repetitions=10, save_path="results_typing.json"
):
    """
    Run the loso_evaluate experiment multiple times, calculate the average and standard deviation
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


# RESULTS
print("RESULTS")
# run_multiple_tremor_experiments(sdataset, k=5, n_repeats=5, repetitions=10)
run_multiple_typing_experiments(sdataset, repetitions=10)
