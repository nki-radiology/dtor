from dtor.datasets.dataset_nominal import CTImageDataset

dset = CTImageDataset("data/external/Label.csv", label="L_LTP_date",
        shape=(112,112,16), stride=(56, 56, 8))
#shape=(64,64,48), stride=(32, 32, 24))
