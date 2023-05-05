#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def load_data(data_dir, partition, dataset):
    all_data = []
    all_label = []
    if(dataset == "ModelNet40"):
      for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_*.h5'.format(partition))):
        f = h5py.File(h5_name)
        print(f)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
      all_data = np.concatenate(all_data, axis=0)
      all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_data(data_dir, partition, dataset):
    all_data = []
    all_label = []
    if(dataset == "ModelNet10")    
      for h5_name in glob.glob(os.path.join(data_dir, 'modelnet10', 'modelnet10_{}_*.h5'.format(partition))):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

class ModelNet10(Dataset):
    def __init__(self, data_dir, num_points, partition='train'):
        self.data, self.label = load_data(data_dir, partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return

class ModelNet40(Dataset):
    def __init__(self, data_dir, num_points, partition='train'):
        self.data, self.label = load_data(data_dir, partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    data_dir = '/home/nkatta9/Desktop/CNN'
    train = ModelNet40(data_dir, 1024)
    test = ModelNet40(data_dir, 1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)








