# -*- coding: utf-8 -*-
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import math


class LinRegression:
    def __init__(self, X,Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.n = len(self.X)
        self.x_mean = np.mean(self.X)
        self.y_mean = np.mean(self.Y)
        self.b1 = (self.X.dot(self.Y).sum() * self.n - self.X.sum() * self.Y.sum())/ (np.square(self.X).sum()*self.n - pow(self.X.sum(),2))
        self.b0 = self.y_mean - self.b1 * self.x_mean
        self.Y_pred = self.f_reg(self.X)
        print('b0, b1: ',self.b0, self.b1)
        self.metrics()
        self.graph()

    def f_reg(self, x):
        return self.b0  + self.b1 * x

    def predict(self, x):
        return self.f_reg(x)

    def graph(self):
        x = np.linspace(np.min(self.X), np.max(self.X))
        plt.plot(self.X, self.Y, 'ro')
        plt.plot(x, self.f_reg(x))
        plt.show()

    def metrics(self):
        E = math.sqrt(np.sum(np.square(self.Y-self.Y_pred)) / (self.n - 2))
        Q = np.sum(np.square(self.Y-self.y_mean))
        Qr =np.sum(np.square(self.Y_pred-self.y_mean))
        Qe =np.sum(np.square(self.Y-self.Y_pred))
        r2 = Qr / Q
        r = math.sqrt(r2) * math.copysign(1, self.b1)
        print('Стандартная ошибка: ', E)
        print('Q, Qr, Qe: ', Q, Qr, Qe)
        print('r^2, r: ', r2, r)

if __name__ == '__main__':
    data = genfromtxt('data/dataForReg.csv', delimiter=',', skip_header=1).astype(np.int)
    print('Data: ')
    print(data)
    X = [i[0] for i in data]
    Y = [i[1] for i in data]
    reg = LinRegression(X, Y)
    print(reg.predict(13))
