import pickle as pkl
import scipy.signal
import numpy as np


def unpickle_data(sfilepath, gfilepath):
    # read sdata
    with open(sfilepath, "rb") as f:
        tremor_sdata = pkl.load(f)

    # read gdata
    with open(gfilepath, "rb") as f:
        tremor_gdata = pkl.load(f)

    return tremor_sdata, tremor_gdata


def calculate_energy(segment, fs=100, band=None):
    if band is None:
        band = [3, 7]
    energies = []
    for axis in range(segment.shape[1]):
        f, Pxx = scipy.signal.welch(segment[:, axis], fs=fs)
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

    # Zero-padding if less than Kt segments
    if len(bag) < Kt:
        padding = [np.zeros((500, 3)) for _ in range(Kt - len(bag))]
        bag.extend(padding)

    return np.array(bag)
