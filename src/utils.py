import pickle as pkl
import scipy.signal
import numpy as np
import pandas as pd


def unpickle_data(sfilepath, gfilepath):
    # read sdata
    with open(sfilepath, "rb") as f:
        tremor_sdata = pkl.load(f)

    # read gdata
    with open(gfilepath, "rb") as f:
        tremor_gdata = pkl.load(f)

    return tremor_sdata, tremor_gdata


def calculate_energy(segment, fs=100, nperseg=500, band=None):
    if band is None:
        band = [3, 7]
    energies = []
    for axis in range(segment.shape[1]):
        f, Pxx = scipy.signal.welch(segment[:, axis], fs=fs, nperseg=nperseg)
        band_energy = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
        energies.append(band_energy)
    return np.sum(energies)  # Sum energy across all axes


def create_bag(subject, E_thres, Kt):
    bag = []
    for session in subject[3]:
        filtered_session = [segment for segment in session if calculate_energy(segment) > E_thres]
        if len(filtered_session) >= 2:
            bag.extend([segment for segment in filtered_session])
    bag.sort(key=lambda segment: calculate_energy(segment), reverse=True)
    bag = bag[:Kt]

    if len(bag) < min(30, Kt):
        return None, None

    # Zero-padding if less than Kt segments
    if len(bag) < Kt:
        bag_mask = [1] * len(bag) + [0] * (Kt - len(bag))
        padding = [np.zeros((500, 3)) for _ in range(Kt - len(bag))]
        bag.extend(padding)
    else:
        bag_mask = [1] * Kt

    return np.array(bag), np.array(bag_mask)


def form_dataset(tremor_data, E_thres, Kt):
    data = []
    mask = []
    for subject_id in tremor_data.keys():
        # Check if the subject's annotations are valid
        if isinstance(tremor_data[subject_id][1], dict) and 'tremor_manual' in tremor_data[subject_id][1]:
            # Create the bag for the subject
            bag, bag_mask = create_bag(tremor_data[subject_id], E_thres, Kt)

            # Get the associated label
            label = tremor_data[subject_id][1]['tremor_manual']

            # Append the tuple (X, y) to the data list
            if bag is not None:
                data.append((bag, label))
                mask.append(bag_mask)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data, columns=['X', 'y'])
    mask = np.array(mask)

    return df, mask
