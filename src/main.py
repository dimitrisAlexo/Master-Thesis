from preprocessing import *

plt.style.use("ggplot")


start = time.time()

# Adjust the paths to be relative to the current script location
sdata_path = os.path.join('..', 'data', 'tremor_sdata.pickle')
gdata_path = os.path.join('..', 'data', 'tremor_gdata.pickle')
tremor_sdata, tremor_gdata = unpickle_data(sdata_path, gdata_path)

E_thres = 0.15
Kt = 1500
sdataset = form_dataset(tremor_sdata, E_thres, Kt)

print(sdataset)

print(time.time() - start)

