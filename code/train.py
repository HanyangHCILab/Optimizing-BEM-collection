import argparse
import torch
from model.custom.CNN import CNN
from model.custom.GCN import ST_GCN
from model.custom.LSTM import CNNLSTMModel
from model.custom.Transformer import coatnet_0
import torch.nn as nn  # 신경망들이 포함됨

def select_model(model_type):
    if(model_type == "CNN"):
        return CNN()
    elif(model_type == "GCN"):
        return ST_GCN(num_class=7,
                    in_channels=3,
                    edge_importance_weighting= False,
                    )
    elif(model_type == "LSTM"):
        return CNNLSTMModel()
    elif(model_type == "Transformer"):
        return coatnet_0()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', help='checkpoint directory', default = "None")
    parser.add_argument('-m', '--model_type', default='CNN', type=str, help='type of model: CNN, GCN, LSTM, Transformer')
    parser.add_argument('-e', '--evaluate',  type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)', default = "False")
    parser.add_argument('-d', '--data_type',  type=str, metavar='FILENAME', help='type_of_data: mocap, kinect, iphone, 3DPE',default = "mocap")
    parser.add_argument('-p', '--predefined',  type=str, metavar='FILENAME', help='custom or predefined',default = "predefined")

    opts = parser.parse_args()
    return opts

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
from data_processing import scaling_data, transfer_input_shape


def load_data(data_type):
    train_data_path = "../data/device/train/" +  f"{data_type}_train.npy"
    train_label_path = "../data/device/train/" +  f"{data_type}_train_label.npy"
    test_data_path = "../data/device/test/" +  f"{data_type}_test.npy"
    test_label_path = "../data/device/test/" +  f"{data_type}_test_label.npy"
  
    X_train = np.load(train_data_path)
    y_train = np.load(train_label_path)
    X_test = np.load(test_data_path)
    y_test = np.load(test_label_path)

    return (X_train, y_train, X_test, y_test)

def preprocess_data(dataset,model_type):

    X_train,y_train,X_test,y_test = dataset
    
    joints = 16
    X_train, scaler = scaling_data(X_train,joints)
    X_test, _ = scaling_data(X_test,joints,scaler)

    X_train, y_train = transfer_input_shape(X_train,y_train,model_type)
    X_test, y_test = transfer_input_shape(X_test,y_test,model_type)
    
    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_test,  y_test)
    return train_set, val_set
    
import os
import time
from scheduler import CosineAnnealingWarmUpRestarts
from sklearn.metrics import f1_score, accuracy_score

def select_optimizer(model_type, model):
    if model_type == "CNN":
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                                                lr_lambda=lambda epoch: 0.95 ** epoch)
    elif model_type == "LSTM":
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                                                lr_lambda=lambda epoch: 0.95 ** epoch)
    elif model_type == "GCN":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay = 0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif model_type == "Transformer":
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=0.001,  T_up=5, gamma=0.5)
    return optimizer, scheduler

def train(args):
    model_type, data_type = args.model_type, args.data_type
    # prepare data
    dataset = load_data(data_type)
    train_set, val_set = preprocess_data(dataset,model_type)

    # set dataloader
    BATCH_SIZE = 64
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    # select model type
    device = "cuda"
    if(args.predefined == "custom"):
        model = select_model(model_type).to(device)
        torch.save(model,f"model/{model_type}.pt")
    else:
        model = torch.load(f"model/{model_type}.pt")

    # save model
    if not os.path.exists(f"../pretrained_models/{model_type}"):
        os.mkdir(f"../pretrained_models/{model_type}")

    # set hyperparameter (Lr, optimizer, scheduler)
    NUM_EPOCH = 80  # SET EPOCH
    LEARNING_RATE = 0.001
    TOTAL_BATCH = len(train_loader)
    VAL_TOTAL_BATCH = len(val_loader)

    criterion = nn.CrossEntropyLoss().to(device)  # Loss
    optimizer, scheduler = select_optimizer(model_type,model) # Optimizer & Scheduler

    # train model
    
    model.train()

    for epoch in range(1,NUM_EPOCH+1):  # repeat as epochs
        avg_cost = 0
        correct = 0
        total = 0
        start_time = time.time()

        model.train()

        for batch_idx, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()  # set model's gradient 0
            out = model(X)  # forward propagation
            cost = criterion(out, Y)  # output & target loss 
            Y_pred = torch.max(out.data, 1)[1]
            total += len(Y)  # The number of total data

            cost.backward()  # backward propagation and calculate gradients
            optimizer.step()  # update parameters

            avg_cost += cost / TOTAL_BATCH  # calculate training loss
            correct += (Y_pred == Y).sum().item()  # The number of correct answer

        if(epoch%1 ==0):
            # print training accuracy per every epoch
            print(f"================ epoch: {epoch} ===================")
            print('[Epoch: {:>4}, Time: {:>0.2f}] train_acc = {:>0.2f} %, cost = {:>.9}'.format(epoch , time.time() - start_time, 100. * correct / total, avg_cost))

        # switch to evaluate mode (do not calculate gradient & update parameters )
        model.eval()
        val_correct,val_avg_cost,val_total = 0,0,0
        pred_list = []
        label_list = []
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(val_loader):
                X = X.to(device)
                Y = Y.to(device)

                out = model(X)
                Y_pred = torch.max(out.data, 1)[1]  # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
                pred_list.append(Y_pred) # predict label
                label_list.append(Y) # valid label

                val_total += len(Y)  # total number of data
                val_correct += (Y_pred == Y).sum().item()  # 
                val_cost = criterion(out, Y)
                val_avg_cost += val_cost / VAL_TOTAL_BATCH


            y_pred = torch.concat(pred_list).detach().cpu().numpy()
            y_true = torch.concat(label_list).detach().cpu().numpy()

            avg_f1_score = f1_score(y_true, y_pred, average='macro')
            if(epoch%1 ==0):
                print('Test Accuracy: {:0.2f} %'.format(100. * val_correct / val_total))
        scheduler.step()

    torch.save(model.state_dict(), f'../pretrained_models/{model_type}/{data_type}_UE_state_dict_{NUM_EPOCH}'+'.pt')


def evaluate(args):
    
    devices = ['mocap','3DPE','kinect','iphone']
    models = ['CNN','LSTM','GCN','Transformer']
    tables = np.zeros((4,4))

    for i, data_type in enumerate(devices):
        for j, model_type in enumerate(models):
            # load train data
            dataset = load_data(data_type)

            # preprocessing data (scaling, transform input)
            _, val_set = preprocess_data(dataset,model_type)
            val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

            # load pretrained model
            device = 'cuda'
            
            if(args.predefined == "predefined"):
                model = torch.load(f"model/{model_type}.pt")
            else:
                model = select_model(model_type).to(device)

            model_state_dict = torch.load(f"../pretrained_models/{model_type}/{data_type}_UE_state_dict_80.pt", map_location=device)
            model.load_state_dict(model_state_dict)

            #evaluate
            model.eval()
            val_correct,val_avg_cost,val_total = 0,0,0
            pred_list = []
            label_list = []
            with torch.no_grad():
              for batch_idx, (X, Y) in enumerate(val_loader):
                  X = X.to(device)
                  Y = Y.to(device)

                  out = model(X)
                  Y_pred = torch.max(out.data, 1)[1]  # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출

                  pred_list.append(Y_pred)
                  label_list.append(Y)
              y_pred = torch.concat(pred_list).detach().cpu().numpy()
              y_true = torch.concat(label_list).detach().cpu().numpy()

              f1_score_ = f1_score(y_true, y_pred, average='micro')
              tables[i,j] = f1_score_
              print(f1_score_)
        
    tables = pd.DataFrame(data = tables, columns = models)
    tables.index = devices
    print(tables)
    return 0


def run():
    args = parse_args()
    if (args.evaluate == "False"):
        train(args)
    else:
        evaluate(args)

if __name__ == '__main__':
    run()