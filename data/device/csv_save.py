import numpy as np
import pandas as pd


devices = ['mocap','3DPE','kinect','iphone']
people_index = np.load("UE_index.npy")


for data_type in devices:
    data = np.load(f"{data_type}.npy")
    label = np.load("device_label.npy")
    start_index = 0
    end_index =0
    for i,index in enumerate(people_index):
        trial_index = np.zeros((7))
        end_index = start_index + index
        arr_1p = data[start_index:end_index]
        label_1p = label[start_index:end_index]
        for j,arr_1t in enumerate(arr_1p):
            arr_1t = pd.DataFrame(arr_1t)
            trial = trial_index[label_1p[j]] +1
            trial_index[label_1p[j]] = trial_index[label_1p[j]] +1 
            arr_1t.to_csv(f"csv/{data_type}/{i+1}_{label_1p[j]+1}_{int(trial)}.csv")