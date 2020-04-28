
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import seaborn as sns
import matplotlib
import scipy.io.wavfile as wav
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from scipy.signal import spectrogram
 # interactive mode
timit_dir = r'C:\Users\YAbraham\PycharmProjects\pytorch\audio\timit\darpa-timit-acousticphonetic-continuous-speech\data'
from scipy import signal as sig
import pickle
import seaborn as sns
from librosa.feature import mfcc, delta
import librosa.display as display

from sklearn.decomposition import PCA

# t - > read files - >
from preprocess import *


class Filter(object):
    def __init__(self):
        self.minSigSize = 500
        self.remove_labels = ['h#']

    def __call__(self, data):
        if type(data) != type(pd.DataFrame()):
            data = pd.DataFrame(data)
        for lbl in self.remove_labels:
            data = self.remove_label(lbl,data)

        data = self.remove_by_sig_len(data)
        return data

    def remove_by_sig_len(self,data):
        data['sigLen'] = data['end'] - data['start']
        data = data[data['sigLen'] > self.minSigSize]
        return data

    def remove_label(self,lbl,df):
        data = df[df['lbl'] != lbl]
        return data


class ReadFiles(object):

    def __init__(self):
        self.get_sig_data  =False

    def __call__(self,pack):
        self.pack = pack
        return self.phn_data

    def __len__(self):
        return self.len
    @property
    def phn_data(self):
        pnm  = self.read_phn()
        phn_data = []
        if self.get_sig_data:
            rate,samp = self.read_wav()
            for p in pnm:
                phn_data.append( {'samp' : samp[p['start']:p['end']], 'lbl': p['lbl'],'rate' : rate})
            self.len = len(phn_data)
        return pnm

    def read_wav(self):
        try:
            return wav.read(self.pack['wav'])
        except:
            pass
    def read_phn(self):
        li = []
        with open(self.pack['phn'],'r') as phn:
            line = phn.readline()
            while line:
                sp = line.split(' ')
                p = {'start': int(sp[0]), 'end': int(sp[1]), 'lbl': sp[2].split('\n')[0],'wav_directory' :self.pack['wav']}
                li.append(p)
                line = phn.readline()

        return li

    #def read_wrd(self):

    #def read_text(self):


class TIMIT(Dataset):
    """ read timit dataset"""
    """ <USAGE>/<DIALECT>/<SEX><SPEAKER_ID>/<SENTENCE_ID>.<FILE_TYPE>"""

    def __init__(self,dir , transform = None,quiry ='TRAIN'):

        self.Filter = Filter()
        self.Read = ReadFiles()
        self.dir = dir
        self.transform = transform
        self.RD = os.path.basename(dir)
        self.TEST_data_dir = []
        self.TRAIN_data_dir = []
        self.TRAIN_data= []
        self.TEST_data = []
        self.quiry = quiry #'TRAIN'  # 'TEST'
        if os.path.basename(self.dir) == 'TIMIT.pkl':
            with open(self.dir,'rb')  as input:
                tmt = pickle.load(input)
                self.TRAIN_data = tmt.TRAIN_data
                self.TEST_data = tmt.TEST_data
                self.TEST_data_filtered = self.Filter(self.TEST_data)
                self.TRAIN_data_filtered = self.Filter(self.TRAIN_data)
                self.TEST_data_dir =  tmt.TEST_data_dir
                self.TRAIN_data_dir =  tmt.TEST_data_dir


        else:
            self.USAGE()


    def USAGE(self):

        test = os.path.join(self.dir, 'TEST')
        self.DIALECT(dir =test,use = 'TEST')
        train =os.path.join(self.dir, 'TRAIN')
        self.DIALECT(dir = train, use = 'TRAIN')

    def DIALECT(self,dir,use):
        DR = glob.glob(os.path.join(dir,'*'))
        for D in DR:
            self.SPEAKER(dir = D, DR = os.path.basename(D),use = use)

    def SPEAKER(self,dir,DR,use):
        spkr = glob.glob(os.path.join(dir, '*'))
        for s in spkr:
            sp = os.path.basename(s)
            self.pack(dir = s,speaker = sp,DR = DR,use = use)

    def pack(self,dir,speaker,DR,use):


        wavfiles = glob.glob(os.path.join(dir, '*.WAV.wav'))
        for wav in wavfiles:
            nm = os.path.basename(wav).split('.')[0]
            phn = os.path.join(dir, '{}.PHN'.format(nm))
            wd = os.path.join(dir, '{}.WRD'.format(nm))
            txt =  os.path.join(dir, '{}.TXT'.format(nm))
            dict = {'use': use, 'DR': DR, 'sex': speaker[0], 'speaker_ID': speaker[1::], 'sentence_ID': nm,
                    'wav': wav, 'phn': phn, 'wd': wd, 'txt': txt}
            if use == 'TEST':
                self.TEST_data_dir.append(dict)
            if use == 'TRAIN':
                self.TRAIN_data_dir.append(dict)

        self.__read_all(use)

    def save(self):
        saveDir = os.path.join(self.dir, 'TIMIT.pkl')
        with open (saveDir,'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def __read_all(self,use):

        if use == 'TEST':
            self.TEST_data = pd.DataFrame(sum([self.Read(p) for p in self.TEST_data_dir], []),index=None)
            self.TEST_data_filtered = self.Filter(self.TEST_data)
        if use == 'TRAIN':
            self.TRAIN_data = pd.DataFrame(sum([self.Read(p) for p in self.TEST_data_dir], []),index=None)
            self.TRAIN_data_filtered = self.Filter(self.TRAIN_data)

    def __len__ (self):
        if self.quiry == 'TEST':
            return len(self.TEST_data_filtered)
        if self.quiry == 'TRAIN':
            return len(self.TRAIN_data_filtered)

    def __read_pnm(self,pnm_dict):

        rate,samp = wav.read(pnm_dict['wav_directory'])
        return {'samp' : samp[pnm_dict['start']:pnm_dict['end']], 'lbl': pnm_dict['lbl'],'rate' : rate}

    def __getitem__(self, idx):
        if self.quiry == 'TEST':
            samp =  self.TEST_data_filtered.iloc[[idx]].to_dict('records')[0]
        if self.quiry == 'TRAIN':
            samp =  self.TRAIN_data_filtered.iloc[[idx]].to_dict('records')[0]


        samp = self.__read_pnm(samp)
        if self.transform:
            samp = self.transform(samp)

        return samp




#
# ff = ReadFiles()
# s =Spectograph()
# all = []
# i=0
# for nn in t:
#     all+=(s(ff(nn),plot = False))
#     i+=1
#
#     print(i," : ",len(all))
#
# with open(r'C:\Users\YAbraham\Desktop\School\lab rand var\TIMIT_pone_py_data\timit_data_list_test_mfcc.pkl', 'wb') as output:
#
#     pickle.dump(all, output, pickle.HIGHEST_PROTOCOL)

def find_max_spec_size(filelist):
    s0 = 0
    s1 = 1
    remove = []
    keep = []
    labels = []
    i=0
    for f in filelist:
        with open(f,'rb') as input:
            li = pickle.load(input)
        for el in li:
            i+=1
            print(i)
            if el['lbl'] != 'h#':
                if el['spec'] is not None:
                    ss0,ss1 = el['spec'].shape
                    if ss0 > s0:
                        s0 = ss0
                    if ss1 > s1:
                        s1 = ss1

                else:
                    remove.append(el)
            else:
                remove.append(el)
            labels.append(el['lbl'])

    remLbl = [l['lbl'] for l in remove]

    return remLbl,labels,s0,s1

def get_all(li):
    for l in li:
        l['size'] = len(l['samp'])
    df = pd.DataFrame(li)
    df = df[df['lbl'] != 'h#']
    return df



def plot_all(li):
    lb = []

    fig, ax = plt.subplots()
    for l in li.keys():
        if l != '#h':
            ax = sns.distplot(li[l],kde = False,ax=ax)
            lb += [l]
    ax.legend(lb)

def bar_plot(df):
    ax,fig = plt.subplot()
    df = df.sort_values(by=['size'])
    sns.barplot(x='lbl', y='size', data=df,ax=ax).set(title='size of phoneme')

transform = transforms.Compose([Spectograph(),NormSpectograph(),NormSpectograph(),ToTensor()])
savedTimit = r'C:\Users\YAbraham\PycharmProjects\pytorch\audio\timit\darpa-timit-acousticphonetic-continuous-speech\data\TIMIT.pkl'
t = TIMIT(savedTimit,transform = transform)
tt = t[0]
ff =0
# lii = [r'C:\Users\YAbraham\Desktop\School\lab rand var\TIMIT_pone_py_data\timit_data_list_train.pkl',r'C:\Users\YAbraham\Desktop\School\lab rand var\TIMIT_pone_py_data\timit_data_list.pkl']
# #remLbl,labels,s0,s1 = find_max_spec_size(lii)
# with open(lii[1], 'rb') as input:
#     data = get_all(pickle.load(input))
#

#bar_plot(data)

#
# def get_all_ft(data):
#     n = Normilize_fetures()
#     n.set_size(timitData=data)
#     samps = data['samp'].values
#     ft = []
#     for i,s in enumerate(samps):
#         ft.append(n(s))
#         print(i)
#     data['fcc'] = ft
#
#     return data
#
#
# def get_all_ft2(data):
#     n = Normilize_fetures()
#     n.set_size(timitData=data)
#     norm = lambda ft: n(ft)
#     samps = data['samp'].values
#     func = np.vectorize(norm)
#     allft = func(samps)
#
#
#     data['fcc'] = allft
#
#     return data
# data = get_all_ft(data)
#
# # with open(r'C:\Users\YAbraham\Desktop\School\lab rand var\TIMIT_pone_py_data\timit_data_list.pkl', 'rb') as input:
# #     ompany1 = pickle.load(input)
# sss= 6
#
# # with open(r'C:\Users\YAbraham\Desktop\School\lab rand var\TIMIT_pone_py_data\timit_data_list_test_mfcc.pkl', 'wb') as output:
# #
# #     pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
# #routing enumerate
#
# #remLbl,labels,s0,s1 = find_max_spec_size(lii)
# with open(r'C:\Users\YAbraham\Desktop\School\lab rand var\TIMIT_pone_py_data\timit_data_list_test_mfcc.pkl', 'rb') as input:
#     data = pickle.load(input)