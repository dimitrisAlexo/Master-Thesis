import pickle as pkl


def unpickle_data(sfilepath, gfilepath):
    # read sdata
    with open(sfilepath, "rb") as f:
        tremor_sdata = pkl.load(f)

    # read gdata
    with open(gfilepath, "rb") as f:
        tremor_gdata = pkl.load(f)

    return tremor_sdata, tremor_gdata
