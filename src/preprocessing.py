from utils import *
import time
import os

start = time.time()

# Adjust the paths to be relative to the current script location
sdata_path = os.path.join('..', 'data', 'tremor_sdata.pickle')
gdata_path = os.path.join('..', 'data', 'tremor_gdata.pickle')
tremor_data = unpickle_data(sdata_path)

print(tremor_data.keys())
print(len(tremor_data['ac6a73f3422c2c23'][3][0][0]))  # tremor_sdata[subject][index][call][segment][acceleration]
print(tremor_data['ac6a73f3422c2c23'][1])  # annotation
print(tremor_data['6cc41389d3e9aea9'][1])
print(tremor_data['f78737822482c8e4'][1])
print(tremor_data['4f8329ca79baa7e2'][1])
print(tremor_data['1a2c87aa323414fa'][1])

print(calculate_energy(tremor_data['ac6a73f3422c2c23'][3][0][0]))
print(calculate_energy(tremor_data['6cc41389d3e9aea9'][3][0][0]))
print(calculate_energy(tremor_data['f78737822482c8e4'][3][0][0]))
print(calculate_energy(tremor_data['4f8329ca79baa7e2'][3][0][0]))
print(calculate_energy(tremor_data['1a2c87aa323414fa'][3][0][0]))

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

# print(time.time() - start)
