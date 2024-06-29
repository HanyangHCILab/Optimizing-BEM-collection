import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def scaling_data(data,joints,scaler=None):
    
    width = 3 * joints
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(data.reshape(len(data)*150,width))
    data = scaler.transform(data.reshape(len(data)*150,width)).reshape(-1,150,width)
    data = np.array([data[:,:,0::3],data[:,:,1::3],data[:,:,2::3]])
    return data , scaler

from torchvision import transforms
# input: numpy, output: torch array
def transfer_input_shape(data, label, model_type):

    data = torch.from_numpy(data)
    label = torch.from_numpy(label)

    # batch, channel, height(timeseries), width(joints)
    data = data.permute(1,0,2,3) 

    if(model_type == "GCN"):
        data = data.reshape(*data.shape,1)
    

    if(model_type == "Transformer"):
        data = transforms.Resize((192, 16))(data)

    if(model_type == "LSTM"):
        b,c,t,j = data.shape
        data = data.reshape(-1,5,c,30,j)

    return data, label
