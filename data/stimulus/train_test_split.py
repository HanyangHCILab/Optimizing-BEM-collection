import numpy as np
import pandas as pd
import os

def load_data(data_Type):
    train_data_path =   f"npy/{data_Type}.npy"
    train_label_path =  f"npy/label.npy"
  
    X = np.load(train_data_path )
    y = np.load(train_label_path)


    return X, y

def split_data(X,y):
  

    LOSO_pred_list = []
    LOSO_label_list = []


    X_train = np.concatenate([X[:13*21]],axis=0)
    y_train = np.concatenate([y[:13*21]],axis=0)
    X_test = X[13*21:]
    y_test = y[13*21:]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__": 
    stimulus = ['word','picture','video']

    for stimuli in stimulus:
        data,label = load_data(stimuli)
        X_tr,y_tr,X_te,y_te = split_data(data,label)

        np.save(f"train/{stimuli}_train.npy",X_tr)
        np.save(f"train/{stimuli}_train_label.npy",y_tr)
        np.save(f"test/{stimuli}_test.npy",X_te)
        np.save(f"test/{stimuli}_test_label.npy",y_te)
      