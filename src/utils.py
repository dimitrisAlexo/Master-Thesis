import pickle as pkl
import time

import scipy.signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def unpickle_data(filepath):
    with open(filepath, "rb") as f:
        data = pkl.load(f)

    return data


def calculate_energy(segment, fs=100, nperseg=500, band=None):
    if band is None:
        band = [3, 7]
    energies = []
    for axis in range(segment.shape[1]):
        f, Pxx = scipy.signal.welch(segment[:, axis], fs=fs, nperseg=nperseg)
        band_energy = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
        energies.append(band_energy)
    return np.sum(energies)  # Sum energy across all axes


def calculate_metrics(tn, fp, fn, tp):
    accuracy = safe_metric(tp + tn, tn + fp + fn + tp)
    sensitivity = safe_metric(tp, tp + fn)
    specificity = safe_metric(tn, tn + fp)
    precision = safe_metric(tp, tp + fp)
    f1_score = safe_metric(2 * sensitivity * precision, sensitivity + precision)
    return accuracy, sensitivity, specificity, precision, f1_score


def create_bag(subject, E_thres, Kt):
    bag = []
    for session in subject[3]:
        num_pairs = session.shape[0] // 2
        session = np.concatenate(
            [session[2 * i:2 * i + 2].reshape(1, 1000, 3) for i in range(num_pairs)], axis=0
        )
        filtered_session = [segment for segment in session if calculate_energy(segment) > E_thres]
        if len(filtered_session) >= 2:
            bag.extend([segment for segment in filtered_session])
    bag.sort(key=lambda segment: calculate_energy(segment), reverse=True)
    bag = bag[:Kt]

    if len(bag) < min(10, Kt):
        return None

    # Zero-padding if less than Kt segments
    if len(bag) < Kt:
        padding = [np.zeros((1000, 3)) for _ in range(Kt - len(bag))]
        bag.extend(padding)

    return np.array(bag)


def form_dataset(tremor_data, E_thres, Kt, train_label_str, test_label_str):
    # Set of valid labels
    valid_labels = {'updrs16', 'updrs20', 'updrs21', 'tremor_manual'}
    assert train_label_str in valid_labels, f"train_label_str '{train_label_str}' is not valid."
    assert test_label_str in valid_labels, f"test_label_str '{test_label_str}' is not valid."

    data = []
    for subject_id in tremor_data.keys():
        # Check if the subject's annotations are valid
        if isinstance(tremor_data[subject_id][1], dict):
            # Create the bag for the subject
            bag = create_bag(tremor_data[subject_id], E_thres, Kt)

            # Get the associated label
            train_label = tremor_data[subject_id][1][train_label_str]

            if test_label_str == 'updrs20':
                test_label = 0 if tremor_data[subject_id][1]['updrs20_right'] + tremor_data[subject_id][1][
                    'updrs20_left'] == 0 else 1
            elif test_label_str == 'updrs21':
                test_label = 0 if tremor_data[subject_id][1]['updrs21_right'] + tremor_data[subject_id][1][
                    'updrs21_left'] == 0 else 1
            else:
                test_label = 0 if tremor_data[subject_id][1][test_label_str] == 0 else 1

            # Append the tuple (X, y) to the data list
            if bag is not None:
                data.append((bag, train_label, test_label))

    # Create a DataFrame from the data list
    df = pd.DataFrame(data, columns=['X', 'y_train', 'y_test'])

    with open("sdataset.pickle", 'wb') as f:
        pkl.dump(df, f)

    return df


def plot_sample(sample):
    plt.figure(figsize=(10, 4))
    plt.plot(sample[:, 0], label='X-axis')
    plt.plot(sample[:, 1], label='Y-axis')
    plt.plot(sample[:, 2], label='Z-axis')
    plt.xlabel('Time Steps')
    plt.ylabel('Accelerometer Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def filter_data(subject, E_thres, Kt):
    bag = []
    for session in subject[3]:
        # print("Session init: ", np.shape(session))
        # plot_sample(session[0])
        # plot_sample(session[1])
        # time.sleep(5)
        num_pairs = session.shape[0] // 3
        session = np.concatenate(
            [session[3 * i:3 * i + 3].reshape(1, 1500, 3) for i in range(num_pairs)], axis=0
        )
        # print("Session end: ", np.shape(session))
        # plot_sample(session[0])
        # time.sleep(5)
        filtered_session = [segment for segment in session if calculate_energy(segment) > E_thres]
        if len(filtered_session) >= 2:
            bag.extend([segment for segment in filtered_session])
    bag.sort(key=lambda segment: calculate_energy(segment), reverse=True)
    bag = bag[:Kt]

    if len(bag) < 10:
        return None

    return np.array(bag)


def form_unlabeled_dataset(tremor_gdata, tremor_sdata, E_thres, Kt):
    print("Length of tremor_gdata: ", len(tremor_gdata))
    tremor_gdata = {key: value for key, value in tremor_gdata.items() if key not in tremor_sdata}
    print("Length of tremor_gdata: ", len(tremor_gdata))

    data = []
    counter = 0

    for subject_id in tremor_gdata.keys():
        if isinstance(tremor_gdata[subject_id][1], dict):
            bag = filter_data(tremor_gdata[subject_id], E_thres, Kt)
            if bag is not None:
                counter += 1
                print(counter)
                # if counter > 50:
                #     break
                data.extend(bag)

    print("Counter: ", counter)

    data = np.array(data)

    # Shuffle only the batches, i.e., along the first axis
    indices = np.random.permutation(data.shape[0])
    data = data[indices]

    with open("unlabeled_data.pickle", 'wb') as f:
        pkl.dump(data, f)

    return data


def form_federated_dataset(tremor_gdata, tremor_sdata, E_thres, Kt, num_clients):

    print("Length of tremor_gdata: ", len(tremor_gdata))
    tremor_gdata = {key: value for key, value in tremor_gdata.items() if key not in tremor_sdata}
    print("Length of tremor_gdata: ", len(tremor_gdata))
    federated_data = []
    counter = 0

    # Collect and filter data
    for subject_id in tremor_gdata.keys():
        if isinstance(tremor_gdata[subject_id][1], dict):
            bag = filter_data(tremor_gdata[subject_id], E_thres, Kt)
            if bag is not None:
                counter += 1
                print(f"Processed subject {counter}: {subject_id}")
                print(f"Length of bag: {len(bag)}")
                federated_data.append(np.array(bag))  # Add bag as a NumPy array

        # Stop once we have collected num_clients bags
        if len(federated_data) >= num_clients:
            break

    print("Total subjects processed: ", counter)

    # Verify there are enough clients
    if len(federated_data) < num_clients:
        raise ValueError("Not enough bags to form the specified number of clients.")

    # Save raw data (list of NumPy arrays)
    with open("federated_data.pickle", 'wb') as f:
        pkl.dump(federated_data, f)

    print("Saved federated_data to 'federated_data.pickle'")

    return federated_data


def normalize(data):
    # Find the min and max values for each batch (across time_steps and channels)
    min_val = np.min(data, axis=(1, 2), keepdims=True)  # shape: (batch_size, 1, 1)
    max_val = np.max(data, axis=(1, 2), keepdims=True)  # shape: (batch_size, 1, 1)

    # Apply the normalization formula
    data_normalized = 2 * (data - min_val) / (max_val - min_val) - 1

    return data_normalized


def normalize_window(window):
    # Get the min and max for each window
    min_val = np.min(window)
    max_val = np.max(window)

    # Apply the normalization formula to scale values between -1 and 1
    normalized_window = 2 * (window - min_val) / (max_val - min_val) - 1

    return normalized_window


def normalize_mil(data):
    """
    Normalize MIL data in the shape (bags, batch_size, time_steps, channels)
    using min-max normalization within each bag and batch.
    """
    # Find the min and max values for each bag and batch (across time_steps and channels)
    min_val = np.min(data, axis=(2, 3), keepdims=True)  # shape: (bags, batch_size, 1, 1)
    max_val = np.max(data, axis=(2, 3), keepdims=True)  # shape: (bags, batch_size, 1, 1)

    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = 1e-8
    data_normalized = 2 * (data - min_val) / (max_val - min_val + epsilon) - 1

    return data_normalized


def safe_confusion_matrix(true_labels, predicted_labels):
    """
    Computes the confusion matrix safely and returns TN, FP, FN, TP.
    Handles cases where only one class is present.
    """
    cm = confusion_matrix(true_labels, predicted_labels)

    # If confusion matrix is 1x1 (only one class present), handle accordingly
    if cm.shape == (1, 1):
        if true_labels[0] == 0:
            # All true labels and predicted labels are 0
            tn = cm[0, 0]
            fp = fn = tp = 0
        else:
            # All true labels and predicted labels are 1
            tp = cm[0, 0]
            tn = fp = fn = 0
    elif cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        raise ValueError("Unexpected confusion matrix shape.")

    return tn, fp, fn, tp


def safe_metric(numerator, denominator):
    """Returns a safe division result to avoid ZeroDivisionError."""
    if denominator == 0:
        return np.nan  # Or return 0, depending on your preference
    return numerator / denominator