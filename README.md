# Behavioral-Research-Methodologies-for-bodily-emotion-recognition

The code for optimizing  bodily emotion motion data collection method

### 1. Set Environment
```
* CUDA:  11.6
* Python:  3.7.16
* Pytorch:  1.13.1
* Timm:  0.9.2
```


### 2. Download Data 
* You can download the data from this link : https://zenodo.org/records/12577086
* Transfrom the data from csv to numpy array ( number of data x time series x joints x axes )
* You can do this just execute the python code  data/{expertise | device | stimulus}/train_test_split.py
__ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ __ ____ __ __ __ __ 



### 3. Training the model
* You can train the model by following code
```
python train.py -s {study_name} -m {model_name} -d {data_type} -p {predefined | custom} 
```
* study_name = expertise | device | stimulus
* model_name = LDA | RF | CNN | LSTM | GCN | Transformer
* data_type
  *   expertise: nonactor | actor
  *   device: mocap | 3DPE | kinect | iphone
  *   stimulus: word | picture | video
* predefined : if predefined then use .pt model else use {model}.py code ( you can custom the model architecture )

### 4. Evaluate the model
```
python train.py -s {study_name} -e True
```
> Execution result
> 
![image](https://github.com/HanyangHCILab/Optimizing-BEM-collection/assets/81300282/80748059-2538-4ec6-bccd-b767d9f62708)
