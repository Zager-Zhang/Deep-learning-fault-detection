import os
import torch
import numpy as np 
import pandas as pd
from scipy.io import loadmat
from datasets.MatrixDatasets import dataset
from datasets.matrix_aug import *
from tqdm import tqdm
import pickle
import pywt
from sklearn.model_selection import train_test_split

signal_size = 1024




#1 Undamaged (healthy) bearings(6X)
HBdata = ['K001',"K002",'K003','K004','K005','K006']
label1=[0,1,2,3,4,5]  #The undamaged (healthy) bearings data is labeled 1-9
#2 Artificially damaged bearings(12X)
ADBdata = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']
label2=[6,7,8,9,10,11,12,13,14,15,16,17]    #The artificially damaged bearings data is labeled 4-15
#3 Bearings with real damages caused by accelerated lifetime tests(14x)
RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI04','KI14','KI16','KI17','KI18','KI21']
label3=[18,19,20,21,22,23,24,25,26,27,28,29,30,31]  #The artificially damaged bearings data is labeled 16-29

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
state = WC[0] #WC[0] can be changed to different working states

def  CWT(lenth,data):
    scale=np.arange(1,lenth)
    cwtmatr, freqs = pywt.cwt(data, scale, 'mexh')
    return cwtmatr

#generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''

    data = []
    lab = []
    for i in tqdm(range(len(HBdata))):
        name1 = state+"_"+HBdata[i]+"_1"
        path1=os.path.join('/tmp',root,HBdata[i],name1+".mat")        #_1----->1 can be replaced by the number between 1 and 20
        data1, lab1 = data_load(path1,name=name1,label=label1[i])
        data += data1
        lab  += lab1

    for j in tqdm(range(len(ADBdata))):
        name2 = state+"_"+ADBdata[j]+"_1"
        path2=os.path.join('/tmp',root,ADBdata[j],name2+".mat")        
        data2,lab2 = data_load(path2,name=name2,label=label2[j])
        data += data2
        lab +=lab2

    for k in tqdm(range(len(RDBdata))):
        name3 = state+"_"+RDBdata[k]+"_1"
        path3=os.path.join('/tmp',root,RDBdata[k],name3+".mat")        
        data3, lab3= data_load(path3,name=name3,label=label3[k])
        data +=data3
        lab +=lab3

    return [data,lab]

def data_load(filename,name,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  #Take out the data
    fl = fl.reshape(-1,)
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        w = int(np.sqrt(signal_size))
        x = fl[start:end]
        imgs = x.reshape(w,w)
        data.append(imgs)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
    'train': Compose([
        ReSize(size=1.0),
        Reshape(),
        Normalize(normlize_type),
        Retype(),
    ]),
    'val': Compose([
        ReSize(size=1.0),
        Reshape(),
        Normalize(normlize_type),
        Retype(),
    ])
}
    return transforms[dataset_type]
#--------------------------------------------------------------------------------------------------------------------
class PUSlice(object):
    num_classes = 32
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
            with open(os.path.join(self.data_dir, "PUSlice.pkl"), 'wb') as fo:
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



