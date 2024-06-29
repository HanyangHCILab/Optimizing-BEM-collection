import numpy as np

def pre_process(data):
    # downsampling to 30 HZ
    data = data[:,::4].reshape(-1,150,48)
    data = data.astype(np.float32)
    # Set starting root point (0,0,09)
    data = data - np.tile(data[:,0:1,0:3],(1,150,16))
    return data

devices = ['mocap','3DPE','kinect','iphone']

for device in devices:
    train_data_path =   f"{device}.npy"
    train_label_path =  f"device_label.npy"
  
    X = np.load(train_data_path )
    X = pre_process(X)

    np.save(f"{device}.npy",X)
