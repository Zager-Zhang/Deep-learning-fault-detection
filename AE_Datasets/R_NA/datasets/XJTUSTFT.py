import os
import pandas as pd
import numpy as np
from datasets.MatrixDatasets import dataset
from datasets.matrix_aug import *
from tqdm import tqdm
import pickle
from scipy import signal
from sklearn.model_selection import train_test_split

signal_size=1024

label1 = [i for i in range(0,5)]
label2 = [i for i in range(5,10)]
label3 = [i for i in range(10,15)]

#generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    WC = os.listdir(root)  # Three working conditions WC0:35Hz12kN WC1:37.5Hz11kN WC2:40Hz10kN

    datasetname1 = os.listdir(os.path.join(root, WC[0]))
    datasetname2 = os.listdir(os.path.join(root, WC[1]))
    datasetname3 = os.listdir(os.path.join(root, WC[2]))
    data = []
    lab =[]
    for i in tqdm(range(len(datasetname1))):
        files = os.listdir(os.path.join('/tmp',root,WC[0],datasetname1[i]))
        for ii in [-4,-3,-2,-1]: #Take the data of the last three CSV files
            path1 = os.path.join('/tmp',root,WC[0],datasetname1[i],files[ii])
            data1, lab1 = data_load(path1,label=label1[i])
            data += data1
            lab +=lab1

    for j in tqdm(range(len(datasetname2))):
        files = os.listdir(os.path.join('/tmp',root,WC[1],datasetname2[j]))
        for jj in [-4,-3, -2, -1]:
            path2 = os.path.join('/tmp',root,WC[1],datasetname2[j],files[jj])
            data2, lab2 = data_load(path2,label=label2[j])
            data += data2
            lab += lab2

    for k in tqdm(range(len(datasetname3))):
        files = os.listdir(os.path.join('/tmp',root,WC[2],datasetname3[k]))
        for kk in [-4,-3, -2, -1]:
            path3 = os.path.join('/tmp',root,WC[2],datasetname3[k],files[kk])
            data3, lab3 = data_load(path3,label=label3[k])
            data += data3
            lab += lab3

    return [data, lab]

def  STFT(fl):
    f, t, Zxx = signal.stft(fl, nperseg=64)
    img = np.abs(Zxx) / len(Zxx)
    return img

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = pd.read_csv(filename)
    fl = fl["Horizontal_vibration_signals"]
    fl = fl.values
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        x = fl[start:end]
        imgs = STFT(x)
        data.append(imgs)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
    'train': Compose([
        ReSize(size=0.97),
        Reshape(),
        Normalize(normlize_type),
        Retype(),
    ]),
    'val': Compose([
        ReSize(size=0.97),
        Reshape(),
        Normalize(normlize_type),
        Retype(),
    ])
}
    return transforms[dataset_type]
#--------------------------------------------------------------------------------------------------------------------
class XJTUSTFT(object):
    num_classes = 15
    inputchannel = 1

    def __init__(self, data_dir,normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            list_data = get_files(self.data_dir, test)
            with open(os.path.join(self.data_dir, "XJTUSTFT.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
            return train_dataset, val_dataset


