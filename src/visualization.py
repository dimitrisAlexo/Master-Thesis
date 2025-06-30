import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_roc_curve(y_true, y_pred_probs, model_name="Attention MIL"):
    """
    Plots the ROC curve for a binary classification model.
    """
    # Compute False positive rate, True positive rate and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})"
    )

    # Plot the baseline (random classifier) ROC
    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random Classifier (AUC = 0.50)",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name="Attention MIL"):
    """
    Plots a confusion matrix heatmap for a binary classification model.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plotting the confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14}
    )

    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0.5, 1.5], ["No Tremor", "Tremor"], fontsize=12)
    plt.yticks([0.5, 1.5], ["No Tremor", "Tremor"], fontsize=12)
    plt.show()


def compare_roc_curves(
    true_labels_with_pretrain,
    predicted_probs_with_pretrain,
    true_labels_without_pretrain,
    predicted_probs_without_pretrain,
):
    """
    Plots ROC curves for two models (with and without pretraining).
    """
    # ROC curve for the model with pretraining
    fpr_with_pretrain, tpr_with_pretrain, _ = roc_curve(
        true_labels_with_pretrain, predicted_probs_with_pretrain
    )
    roc_auc_with_pretrain = auc(fpr_with_pretrain, tpr_with_pretrain)

    # ROC curve for the model without pretraining
    fpr_without_pretrain, tpr_without_pretrain, _ = roc_curve(
        true_labels_without_pretrain, predicted_probs_without_pretrain
    )
    roc_auc_without_pretrain = auc(fpr_without_pretrain, tpr_without_pretrain)

    # Plotting both ROC curves
    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr_with_pretrain,
        tpr_with_pretrain,
        color="blue",
        lw=2,
        label=f"With Pretraining (AUC = {roc_auc_with_pretrain:.2f})",
    )
    plt.plot(
        fpr_without_pretrain,
        tpr_without_pretrain,
        color="red",
        lw=2,
        label=f"Without Pretraining (AUC = {roc_auc_without_pretrain:.2f})",
    )

    plt.plot(
        [0, 1], [0, 1], color="gray", lw=2, linestyle="--"
    )  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.show()


# data = np.load("roc_curve_pretraining.npz")
# true_labels_with_pretrain, predicted_probs_with_pretrain = data['true_labels'], data['predicted_probs']
# data = np.load("roc_curve_no_pretraining.npz")
# true_labels_without_pretrain, predicted_probs_without_pretrain = data['true_labels'], data['predicted_probs']
#
# compare_roc_curves(true_labels_with_pretrain, predicted_probs_with_pretrain,
#                    true_labels_without_pretrain, predicted_probs_without_pretrain)
