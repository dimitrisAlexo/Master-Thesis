from utils import unpickle_data, form_tremor_dataset, form_unlabeled_dataset

sdata_path = "../data/typing_sdata.pickle"
gdata_path = "../data/typing_gdata.pickle"

# Load data using unpickle_data
typing_sdata = unpickle_data(sdata_path)
typing_gdata = unpickle_data(gdata_path)

# --- sdata code ---
# Get subject list
subject_list = list(typing_sdata.keys())
print("Number of subjects:", len(subject_list))
print("Subjects:", subject_list)

# Data for first subject
subject_data = typing_sdata[subject_list[5]]

print("Data for first subject:", subject_data)

# The actual data is in the first element of the list
typing_histograms = subject_data[0]

typing_hist_for_first_session = typing_histograms[0]
# typing_hist_for_second_session = typing_histograms[1]

print("Shape of first session histogram:", typing_hist_for_first_session.shape)

ht_hist = typing_hist_for_first_session[:101]  # Hold time histogram
ft_hist = typing_hist_for_first_session[101:]  # Flight time histogram

print("Hold time histogram shape:", ht_hist.shape)
print("Flight time histogram shape:", ft_hist.shape)

print("Sum of hold time histogram values:", ht_hist.sum())
print("Sum of flight time histogram values:", ft_hist.sum())

# Count number of subjects where typing_sdata[subject][0] is not empty and has at least 5 sessions
non_empty_subjects = [
    subject
    for subject in subject_list
    if typing_sdata[subject][0] and len(typing_sdata[subject][0]) >= 5
]
print(
    "Number of subjects with non-empty [0] entry and at least 5 sessions:",
    len(non_empty_subjects),
)

# Print number of sessions for each kept subject
for subject in non_empty_subjects:
    num_sessions = len(typing_sdata[subject][0])
    print(f"Subject {subject} has {num_sessions} sessions")

# Print the 6 values typing_sdata[subject][1:] for each kept subject
for subject in non_empty_subjects:
    values = typing_sdata[subject][1:]
    print(f"Subject {subject} values (typing_sdata[subject][1:]): {values[:6]}")

# --- gdata code ---
# Get subject list for gdata
g_subject_list = list(typing_gdata.keys())
print("Number of gdata subjects:", len(g_subject_list))

# Data for first subject in gdata
g_subject_data = typing_gdata[g_subject_list[0]]

# The actual data is in the first element of the list
g_typing_histograms = g_subject_data[0]

# Count number of subjects where typing_gdata[subject][0] is not empty and has at least 5 sessions
g_non_empty_subjects = [
    subject
    for subject in g_subject_list
    if typing_gdata[subject][0] and len(typing_gdata[subject][0]) >= 5
]
print(
    "gdata: Number of subjects with non-empty [0] entry and at least 5 sessions:",
    len(g_non_empty_subjects),
)

# Find min and max of typing_gdata[subject][1] across all kept subjects
g_values = [typing_gdata[subject][1] for subject in g_non_empty_subjects]
min_value = min(g_values)
max_value = max(g_values)
num_ones = g_values.count(1)
num_zeros = g_values.count(0)
print(f"Min value of typing_gdata[subject][1]: {min_value}")
print(f"Max value of typing_gdata[subject][1]: {max_value}")
print(f"Number of 1s: {num_ones}")
print(f"Number of 0s: {num_zeros}")


# Load tremor data
tremor_sdata = unpickle_data("../data/imu_sdata.pickle")
tremor_gdata = unpickle_data("../data/imu_gdata.pickle")

# Count common keys between kept typing_sdata subjects and tremor_sdata
print("length of typing_sdata:", len(typing_sdata.keys()))
print("length of tremor_sdata:", len(tremor_sdata.keys()))
common_sdata_keys = set(non_empty_subjects) & set(tremor_sdata.keys())
print(
    f"Number of common keys in kept typing_sdata subjects and tremor_sdata: {len(common_sdata_keys)}"
)

# Keep only tremor_sdata entries with keys in common_sdata_keys
tremor_sdata = {k: tremor_sdata[k] for k in common_sdata_keys}
E_thres = 2 * 0.15
Kt = 100
# sdataset = form_dataset(tremor_sdata, E_thres, Kt, "tremor_manual", "tremor_manual")
# print(sdataset)

gtremor_subject_list = list(tremor_gdata.keys())
print("Number of tremor_gdata subjects:", len(gtremor_subject_list))
# print(tremor_gdata[gtremor_subject_list[2]][0])

# Count common keys between kept typing_gdata subjects and tremor_gdata
common_gdata_keys = set(g_non_empty_subjects) & set(tremor_gdata.keys())
print(
    f"Number of common keys in kept typing_gdata subjects and tremor_gdata: {len(common_gdata_keys)}"
)

tremor_gdata = {k: tremor_gdata[k] for k in common_gdata_keys}

# gdataset = form_unlabeled_dataset(tremor_gdata, tremor_sdata, E_thres, Kt)
# print(gdataset)
