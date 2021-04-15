from dtor.datasets.dataset_nominal import CTImageDataset

dset = CTImageDataset("data/external/Label.csv", label="L_LTP_date",
                      shape=(64, 64, 64), stride=(32, 32, 32))  # shape=(64,64,48), stride=(32, 32, 24))
