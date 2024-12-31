from utils import *
import time
import os
import matplotlib.pyplot as plt

start = time.time()

# Adjust the paths to be relative to the current script location
sdata_path = os.path.join('..', 'data', 'tremor_sdata.pickle')
gdata_path = os.path.join('..', 'data', 'tremor_gdata.pickle')
tremor_sdata = unpickle_data(sdata_path)
tremor_gdata = unpickle_data(gdata_path)

E_thres = 0.15
Kt = 100

# sdataset = form_dataset(tremor_sdata, E_thres, Kt, 'tremor_manual', 'tremor_manual')

with open("sdataset.pickle", 'rb') as f:
    print("Loading sdataset...")
    sdataset = pkl.load(f)

sdataset['X'] = sdataset['X'].apply(lambda window: normalize_window(window))

print(sdataset)


def plot_accelerometer_windows(X, subject_idx, save_fig=False):
    """
    Visualizes 100 windows of accelerometer data for the given subject index.

    Parameters:
    - X: The input data containing accelerometer windows [100, 500, 3]
    - subject_idx: The index of the subject in the dataset to visualize
    - save_fig: If True, saves the figure as 'labeled_data_<subject_idx>.png'
    """
    subject_windows = X[subject_idx]  # Access the windows for the subject

    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    fig.suptitle(f'Subject {subject_idx} Windows', fontsize=20)

    for i in range(100):  # Loop over the 100 windows
        ax = axs[i // 10, i % 10]  # Get the subplot (10x10 grid)
        window_data = subject_windows[i]  # Get the specific window [500, 3]

        # Plot each axis (X, Y, Z) for the window
        ax.plot(window_data[:, 0], label='X', color='r', linewidth=0.5)
        ax.plot(window_data[:, 1], label='Y', color='g', linewidth=0.5)
        ax.plot(window_data[:, 2], label='Z', color='b', linewidth=0.5)

        ax.set_title(f'Window {i}')
        ax.axis('off')  # Turn off axis labels for cleaner look

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title

    if save_fig:
        plt.savefig(f'labeled_data_{subject_idx}.png')  # Save the figure if needed

    plt.show()

    return


X = sdataset['X']

# plot_accelerometer_windows(X, 27, save_fig=True)
# plot_accelerometer_windows(X, 16, save_fig=True)
# plot_accelerometer_windows(X, 41, save_fig=True)
# plot_accelerometer_windows(X, 13, save_fig=True)
# plot_accelerometer_windows(X, 0, save_fig=True)
#
# plot_accelerometer_windows(X, 17, save_fig=True)
# plot_accelerometer_windows(X, 42, save_fig=True)
# plot_accelerometer_windows(X, 25, save_fig=True)
# plot_accelerometer_windows(X, 5, save_fig=True)
# plot_accelerometer_windows(X, 1, save_fig=True)
#
# plot_accelerometer_windows(X, 2, save_fig=True)
# plot_accelerometer_windows(X, 4, save_fig=True)
# plot_accelerometer_windows(X, 6, save_fig=True)
# plot_accelerometer_windows(X, 11, save_fig=True)
# plot_accelerometer_windows(X, 12, save_fig=True)
#
# plot_accelerometer_windows(X, 24, save_fig=True)
# plot_accelerometer_windows(X, 36, save_fig=True)
# plot_accelerometer_windows(X, 8, save_fig=True)
# plot_accelerometer_windows(X, 9, save_fig=True)
# plot_accelerometer_windows(X, 10, save_fig=True)

# Initialize the empty DataFrame
windows_dataset = pd.DataFrame(columns=['X', 'y'])


# Function to label windows and add them to windows_dataset
def label_windows(windows_dataset, X, subject_idx, window_indices, label):
    for window_idx in window_indices:
        # Extract the specific window data
        window_data = X[subject_idx][window_idx]

        # Append the window and label to the dataset
        new_row = pd.DataFrame({
            'X': [window_data],  # Storing as list to avoid expansion
            'y': [label]
        })

        # Concatenate the new row to the windows_dataset
        windows_dataset = pd.concat([windows_dataset, new_row], ignore_index=True)

    return windows_dataset


# Load your dataset (df is already defined in your case)
X = sdataset['X']  # Assuming 'X' contains the accelerometer data of all subjects

# 1. From subject 27 add windows with label 0
windows_dataset = label_windows(windows_dataset, X, subject_idx=27,
                                window_indices=[3, 6, 7, 10, 26, 27, 30, 31, 76, 66], label=0)

# 2. From subject 17 add windows with label 1
windows_dataset = label_windows(windows_dataset, X, subject_idx=17,
                                window_indices=[20, 21, 22, 26, 28, 30, 31, 32, 33, 34], label=1)

# 3. From subject 16 add windows with label 0
windows_dataset = label_windows(windows_dataset, X, subject_idx=16, window_indices=[4, 5, 6, 7, 10, 11, 12, 27, 28, 33],
                                label=0)

# 4. From subject 42 add windows with label 1
windows_dataset = label_windows(windows_dataset, X, subject_idx=42,
                                window_indices=[2, 4, 5, 6, 9, 13, 14, 15, 53, 64], label=1)

# 5. From subject 41 add windows with label 0
windows_dataset = label_windows(windows_dataset, X, subject_idx=41,
                                window_indices=[1, 2, 3, 14, 17, 27, 53, 54, 81, 83], label=0)

# 6. From subject 25 add windows with label 1
windows_dataset = label_windows(windows_dataset, X, subject_idx=25,
                                window_indices=[5, 14, 15, 17, 22, 41, 43, 52, 69, 71], label=1)

# 7. From subject 13 add windows with label 0
windows_dataset = label_windows(windows_dataset, X, subject_idx=13,
                                window_indices=[0, 1, 2, 3, 4, 27, 28, 29, 30, 31], label=0)

# 8. From subject 5 add windows with label 1
windows_dataset = label_windows(windows_dataset, X, subject_idx=5,
                                window_indices=[25, 26, 33, 50, 61, 77, 83, 94, 95, 26], label=1)

# 9. From subject 0 add windows with label 0
windows_dataset = label_windows(windows_dataset, X, subject_idx=0,
                                window_indices=[2, 4, 11, 26, 27, 31, 32, 54, 55, 82], label=0)

# 10. From subject 1 add windows with label 1
windows_dataset = label_windows(windows_dataset, X, subject_idx=1,
                                window_indices=[7, 8, 9, 12, 13, 14, 15, 16, 17, 18], label=1)

windows_dataset = label_windows(windows_dataset, X, subject_idx=2,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=0)

windows_dataset = label_windows(windows_dataset, X, subject_idx=4,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=0)

windows_dataset = label_windows(windows_dataset, X, subject_idx=6,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=0)

windows_dataset = label_windows(windows_dataset, X, subject_idx=11,
                                window_indices=[0, 1, 2, 3, 4, 23, 24, 25, 26, 27], label=0)

windows_dataset = label_windows(windows_dataset, X, subject_idx=12,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=0)

windows_dataset = label_windows(windows_dataset, X, subject_idx=24,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=1)

windows_dataset = label_windows(windows_dataset, X, subject_idx=36,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=1)

windows_dataset = label_windows(windows_dataset, X, subject_idx=8,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=1)

windows_dataset = label_windows(windows_dataset, X, subject_idx=9,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label=1)

windows_dataset = label_windows(windows_dataset, X, subject_idx=10,
                                window_indices=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10], label=1)

# Save the dataset to a CSV file
windows_dataset.to_pickle('labeled_windows_dataset.pickle')

print("Labeled windows dataset has been saved to 'labeled_windows_dataset.pickle'.")

print(windows_dataset)

# print(tremor_data.keys())
# print(len(tremor_data['ac6a73f3422c2c23'][3][0][0]))  # tremor_sdata[subject][index][call][segment][acceleration]
# print(tremor_data['ac6a73f3422c2c23'][1])  # annotation
# print(tremor_data['6cc41389d3e9aea9'][1])
# print(tremor_data['f78737822482c8e4'][1])
# print(tremor_data['4f8329ca79baa7e2'][1])
# print(tremor_data['1a2c87aa323414fa'][1])
#
# print(calculate_energy(tremor_data['ac6a73f3422c2c23'][3][0][0]))
# print(calculate_energy(tremor_data['6cc41389d3e9aea9'][3][0][0]))
# print(calculate_energy(tremor_data['f78737822482c8e4'][3][0][0]))
# print(calculate_energy(tremor_data['4f8329ca79baa7e2'][3][0][0]))
# print(calculate_energy(tremor_data['1a2c87aa323414fa'][3][0][0]))

# subject = [np.array([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
#                     [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
#                     [[0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 0, 0]]]),
#            np.array([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
#                      [[0, 0, 0], [0, 0, 0], [1, 1, 1], [5, 5, 5]],
#                      [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
#            np.array([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
#                      [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
#                      [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])
#            ]
#
# # print(subject)
# print('energy', calculate_energy(np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [5, 5, 5]])))

# E_thres = 0.15
# Kt = 1500
# bag = create_bag(tremor_data['6cc41389d3e9aea9'], E_thres, Kt)
# # {'updrs16', 'updrs20_right', 'updrs20_left', 'updrs21_right', 'updrs21_left', 'tremor_manual'}
# sdataset = form_dataset(tremor_data, E_thres, Kt, 'tremor_manual', 'tremor_manual')
#
# print(sdataset)

# print("Tremor subjects: ", tremor_sdata.keys())
# print(len(tremor_sdata.keys()))
# print("Tremor subjects: ", tremor_gdata.keys())
# print(len(tremor_gdata.keys()))
#
# count = 0
# for key in tremor_gdata.keys():
#     if key in tremor_sdata.keys():
#         print(key)
#         count += 1
#
# print("count: ", count)
#
# print("sdataset subject: ", tremor_sdata['6cc41389d3e9aea9'][3])
# print("gdataset subject: ", tremor_gdata['6cc41389d3e9aea9'][3])

print(time.time() - start)
