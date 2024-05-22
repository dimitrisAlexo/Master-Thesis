from utils import *
import time
import matplotlib.pyplot as plt

start = time.time()

tremor_sdata, tremor_gdata = unpickle_data("data/tremor_sdata.pickle", "data/tremor_gdata.pickle")

print(tremor_sdata.keys())
print(len(tremor_sdata['418d8c4cb78b2514'][3][0][0]))  # tremor_sdata[subject][index][call][segment][acceleration]
print(tremor_sdata['418d8c4cb78b2514'][1])  # annotation



# max_length = max([len(tremor_sdata[subject][3][i]) for subject in tremor_sdata.keys()
#                   for i in range(len(tremor_sdata[subject][3]))])
# print(max_length)

print(time.time() - start)