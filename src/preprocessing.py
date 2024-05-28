from utils import *
import time
import matplotlib.pyplot as plt
import os

start = time.time()

# Adjust the paths to be relative to the current script location
sdata_path = os.path.join('..', 'data', 'tremor_sdata.pickle')
gdata_path = os.path.join('..', 'data', 'tremor_gdata.pickle')
tremor_sdata, tremor_gdata = unpickle_data(sdata_path, gdata_path)

print(tremor_sdata.keys())
print(len(tremor_sdata['418d8c4cb78b2514'][3][0][0]))  # tremor_sdata[subject][index][call][segment][acceleration]
print(tremor_sdata['6cc41389d3e9aea9'][1])  # annotation

segment_energy = calculate_energy(tremor_sdata['418d8c4cb78b2514'][3][0][0])
print(segment_energy)

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

E_thres = 0.15
Kt = 1500
bag = create_bag(tremor_sdata['6cc41389d3e9aea9'], E_thres, Kt)
sdataset = form_dataset(tremor_sdata, E_thres, Kt)

print(sdataset)

print(time.time() - start)
