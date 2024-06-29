import numpy as np
import pandas as pd



def load_data(data_Type):
    train_data_path =   f"{data_Type}.npy"
    train_label_path =  f"device_label.npy"
  
    X = np.load(train_data_path )
    y = np.load(train_label_path)


    return X, y

def split_data(X,y):
    ue = np.load("UE_index.npy")

    LOSO_pred_list = []
    LOSO_label_list = []

    start_index = 0
    end_index= start_index
    for j in range(10):
        end_index += ue[j]

    X_train = np.concatenate([X[:start_index],X[end_index:]],axis=0)
    y_train = np.concatenate([y[:start_index],y[end_index:]],axis=0)
    X_test = X[start_index:end_index]
    y_test = y[start_index:end_index]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__": 
    devices = ['mocap','3DPE','kinect','iphone']

    for device in devices:
        data,label = load_data(device)
        X_tr,y_tr,X_te,y_te = split_data(data,label)

        np.save(f"train/{device}_train.npy",X_tr)
        np.save(f"train/{device}_train_label.npy",y_tr)
        np.save(f"test/{device}_test.npy",X_te)
        np.save(f"test/{device}_test_label.npy",y_te)
