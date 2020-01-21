# -*- coding: utf-8 -*-
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


class BaieseClassificator:
    def __init__(self, data, Y_feature, feature_types):
        self.data = data
        # self.Y = pd.DataFrame(Y)
        self.Y_feature = Y_feature
        self.n_samples = len(data)
        self.features = list(self.data.drop([self.Y_feature], axis=1).columns)
        self.feature_types = feature_types
        self.classes = self.data[self.Y_feature].unique()
        self.classes_count = self.data[self.Y_feature].value_counts().to_dict()
        self.predictions = dict.fromkeys(self.classes, 1)

    def predict(self, x):
        for i in self.classes:
            class_data = self.data[self.data[self.Y_feature] == i]
            class_count = self.classes_count[i]
            P_C = class_count / self.n_samples

            for j in self.features:
                P_XC = 1
                if self.feature_types[j] == 'categorical':
                    P_XC= len(class_data[class_data[j] == x[j]]) / class_count

                else:
                    if self.feature_types[j] == 'continuous':
                        m = class_data[j].mean()
                        s = class_data[j].std()
                        P_XC = (1 / math.sqrt(2 * math.pi * s)) * math.exp(pow(x[j] - m, 2) / (2 * s * s))

                if self.predictions[i] == 1: self.predictions[i] = P_XC
                else: self.predictions[i] *= P_XC

            self.predictions[i] *= P_C

        self.normalize_predictions()
        print('Предсказание для: ', x)
        print(self.predictions)


    def normalize_predictions(self):
        s = sum(self.predictions.values())
        for i in self.predictions:
            self.predictions[i] /= s


if __name__ == '__main__':
    data = pd.read_csv('data/dataForClass.csv')
    print('Data: ')
    print(data.head())
    feature_types = {'Количество звонков': 'categorical',
                     'SМSактивность': 'categorical',
                     'Internetактивность': 'categorical',
                     'Международные звонки': 'categorical',
                     'Уход': 'categorical'}

    x = {'Количество звонков': '<100',
         'SМSактивность': 'Средняя',
         'Internetактивность': 'Средняя',
         'Международные звонки': 'Да'}

    baiese = BaieseClassificator(data, 'Уход', feature_types)
    # baiese.train()
    baiese.predict(x)