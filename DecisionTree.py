# -*- coding: utf-8 -*-
import math
import random
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
import pandas as pd

class DecisionTree:
    def __init__(self, data, Y_feature):
        self.Y_feature = Y_feature
        self.data = data
        self.features = list(data.drop([self.Y_feature], axis=1).columns)
        self.targets = self.data[self.Y_feature]
        self.tree = list([])
        self.nSamples = len(self.data)
        # self.scoreDict = {i : 0 for i in self.features}

        self.F = self.Gain
        self.node = []

        self.tree = self.splitting(self.data, '')
        # print(self.tree)

    def splitting(self, data, name):

        if len(data) == 0:
            print(name)
            print('[]\n')
            return

        if len(data[self.Y_feature].unique()) == 1:
            print(name)
            print(data[self.Y_feature].to_string(), '\n')
            return data[self.Y_feature].unique()[0]

        entropyY = 0
        for i in data[self.Y_feature].unique():
            entropyY -= self.entropy(data[self.Y_feature],i)
        # print('entropyY',entropyY)

        features = list(data.drop([self.Y_feature], axis=1).columns)
        scoreDict = {}
        for i in features:
            scoreDict[i] = self.F(i,entropyY, data)
        # print(scoreDict)
        dropped = max(scoreDict, key=scoreDict.get)

        node = []
        for i in self.data[dropped].unique():
            data_new = data[data[dropped] == i]
            data_new = data_new.drop([dropped], axis = 1)
            name_new = name + ' ' + str(dropped) + ' ' + str(i)
            node.append([(dropped,i), self.splitting(data_new, name_new)])
            # print('node', node)
        return node

    def Gain(self, feature, entropyY, data):
        rem = 0
        for inst in data[feature].unique():
            entropyInst = 0
            temp = data[data[feature] == inst][self.Y_feature]
            for j in temp.unique():
                entropyInst -= self.entropy(temp,j)
            freqInst = self.freq(data[feature],inst)
            rem += entropyInst * freqInst
        return entropyY - rem

    def Ginny(self, feature, entropyY, data):
        rem = 0
        for inst in data[feature].unique():
            entropyInst = 0
            temp = data[data[feature] == inst][self.Y_feature]
            for j in temp.unique():
                entropyInst -= self.freq(temp,j)
            freqInst = self.freq(data[feature],inst)
            rem += entropyInst * freqInst
        return entropyY - rem


    def entropy(self, series, inst):
        freq = self.freq(series, inst)
        return math.log2(freq) * freq

    def freq(self, series, inst):
        return len(series[series == inst])/len(series)


    def predict(self, x):
        self.node = self.tree
        while True:
            if type(self.node) == str or self.node == None:
                return self.node

            for i in self.node:
                feature = i[0][0]
                inst = i[0][1]
                if x[feature] == inst:
                    self.node = i[1]
                    break


if __name__ == '__main__':

    data = pd.read_csv('data/dataForTrees.csv')
    # print(data.head())
    Y_feature = 'VEGETATION'

    tree = DecisionTree(data, Y_feature)


    x = {'STREAM':'true',
         'SLOPE':'steep',
         'ELEVATION':'high'}
    print('\n',x)
    print(tree.predict(x))