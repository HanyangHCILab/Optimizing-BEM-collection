# Optimizing-BEM-collection
The code for optimizing  bodily emotion motion data collection method

### 1. Set Environment
```
* CUDA:  11.6
* Python:  3.7.16
* Pytorch:  1.13.1
* Timm:  0.9.2
```
You can simply install the necessary libraries using the following command:
```
conda env create -f environment.yml
```

### 2. Download Data & Pertrained model
* You can download the data from this link : https://zenodo.org/records/12577086
* Transfrom the data from csv to numpy array ( number of data x time series x joints x axes ) and save it as (mocap.npy | 3DPE.npy | kinect.npy | iphone.npy) in data/device directory
* Label should be saved in the form of sparse int ( 0 ~ 6 ) and the name should be "device_label.npy"
__ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ 
* You can download the pretrained pytorch model from this link: https://drive.google.com/drive/folders/1lYuZZNhmk6-fLwA-HJp-jCrAF5r5BisH?usp=drive_link
* Save it to pretrained/(CNN/LSTM/GCN/Transformer) directory


### 3. Training the model
* You can train the model by following code
```
python train.py -m {model_name} -d {data_type} -p {predefined | custom} 
```
* model_name = CNN | LSTM | GCN | Transformer
* data_type  = mocap | 3DPE | kinect | iphone
* predefined : if predefined then use .pt model else use {model}.py code ( you can custom the model architecture )

### 4. Evaluate the model
```
python train.py -e True
```