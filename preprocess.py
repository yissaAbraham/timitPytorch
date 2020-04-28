

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



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, pnm):
        if type(pnm) == type([]):
            for p in pnm:
                p['spec'] = torch.from_numpy(p['spec'])
        else:
            pnm['spec'] = torch.from_numpy(pnm['spec'])
        s1,s2 = pnm['spec'].shape
        pnm['spec'] = pnm['spec'].reashape((1, s1, s2))
        return pnm

class NormSpectograph(object):
    def __init__(self,shape = (81,81)):
        self.shape = shape
        self.z = np.zeros(self.shape)
    def __call__(self,pnm):
        pnm = self.norm_size(pnm)
        pnm = self.normilize_amp(pnm)
        return pnm
    def normilize_amp(self,pnm):
        if (pnm['spec'].max() - pnm['spec'].min()) != 0:
            pnm['spec'] = (pnm['spec'] - pnm['spec'].min())/(pnm['spec'].max() - pnm['spec'].min())

        else:
            print( 'spectrogram is empty')
        return pnm

    def norm_size(self,pnm):
        try:
            if (pnm['spec'].shape[0] < self.shape[0] or pnm['spec'].shape[1] < self.shape[1]):
                pnm['spec'] = np.concatenate([pnm['spec'], self.z], axis=1)
            pnm['spec'] = pnm['spec'][0:self.shape[0], 0:self.shape[1]]
        except:
            pass
        return pnm



class MFCC_Image(object):
    def __init__(self):
        self.size = 30000
        self.imageLen = 36
        self.mfcc_fts = 12

    def to_size(self,signal):
        mul = self.size / signal.size + 1
        signal=np.tile(signal,int(mul))[0:self.size]
        return signal

    def set_size(self,timitData):
        self.size = timitData['size'].max()
        self.rate = timitData['rate'][1]

    def pca(self,mfcc):
        S, D, DD = mfcc
        pca = PCA(n_components=self.imageLen)
        S =  pca.fit_transform(S)
        D = pca.fit_transform(D)
        DD = pca.fit_transform(DD)
        return S,D,DD

    def mfcc_features(self,signal):
        signal = signal*1.0
        S = mfcc(y=signal,sr = self.rate,n_mfcc=self.mfcc_fts,hop_length=64)
        D = delta(data=S,order=1)
        DD = delta(data=S,order=2)
        return [S,D,DD]

    def __call__(self, samp):
        samp = self.to_size(samp)
        mfc = self.mfcc_features(samp)
        pca = self.pca(mfc)

        return pca


class checkShape(object):
    def __init__(self,shape = (65,65),type = 2):
        self.shape = shape
    def __call__(self,sample):
        shape = sample['image'].shape
        if shape != self.shape:
            raise Exception(
                '"{}" image is in the worng shape "{}" data mustbe "{}"'.format(sample['name'],shape,self.shape))
            return None
        else:
            return sample



class ToTensor(object):
    def __init__(self):
        self.image_name = 'spec'
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, pnm):
        image, label = pnm['spec'], pnm['lbl']

        return {'image': torch.from_numpy(image),
                'label': label}

class Spectograph(object):
    def __init__(self):
        self.plot = False

    def __call__(self,pnm):
        if type(pnm) == type([]):
            for p in pnm:
                #print(p['samp'].size,p['lbl'])
                p['frq'], p['time'], p['spec'] = self.spectrogram(x=p['samp'], fs=p['rate'])
            if self.plot:
                for pp in p:
                    plt.imshow(p['spec'])
                    plt.title(p['lbl'])
        else:
            pnm['frq'],pnm['time'],pnm['spec'] =  self.spectrogram(x = pnm['samp'],fs = pnm['rate'])
            if self.plot:
                io.imshow(pnm['spec'])
        return pnm

    def spectrogram(self,x, fs):
        window_size = int(fs / 100)  # 10 ms
        overlap = window_size / 2  # 5 ms
        data = np.concatenate([x, np.zeros(int(window_size))])
        window = sig.tukey(M=window_size, alpha=0.25)
        try:
            freq, time, spectrogram = sig.spectrogram(x, fs=fs, window=window, nperseg=window_size, noverlap=overlap)
        except:
            return None, None, None
        return freq, time, np.log10(spectrogram)

