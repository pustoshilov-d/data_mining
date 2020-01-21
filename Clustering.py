# -*- coding: utf-8 -*-
import random
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA

class clustering:
    def __init__(self, data: np.array, k = 2, L_type = 1, means_or_median = 'means'):
        self.data = data
        self.k = k
        self.n_features = len(self.data[0])
        self.L = self.l1 if L_type == 1 else self.l2
        self.center = self.means_center if means_or_median == 'means' else self.median_center
        self.clusters = []
        self.centroids = np.array([0.0]*self.k*self.n_features).reshape(self.k,self.n_features)
        self.centroids_new = np.array([0.0] * self.k * self.n_features).reshape(self.k, self.n_features)

        self.train()

    def train(self):
        #инициализация центроидов рандомными элементами, семенами
        self.centroids = self.get_random_k(self.data)
        epoch = 0

        while True:
            self.clusters = [[] for i in range(self.k)]

            #добавление элемента в кластер с минимальным расстоянием от него
            for i in self.data:
                min = int(np.argmin(self.L(self.centroids - i)))
                self.clusters[min] = self.cluster_append(self.clusters[min], i)

            #обновление центроидов
            for i in range(self.k):
                self.centroids_new[i] = self.center(self.clusters[i])
            #есть ли изменения в центроидах?
            if np.all(self.centroids_new == self.centroids): break
            else: self.centroids = self.centroids_new
            epoch += 1

        print('\nEpoch: ', epoch)
        for i in range(self.k):
            print('\nCluster: ', i)
            print(self.clusters[i])
        print('\nCentroids')
        print(self.centroids_new)

    def cluster_append(self, cluster, item):
        return np.append(cluster, item).reshape(len(cluster)+1, self.n_features)

    def l2(self, array):
        return np.array([LA.norm(i) for i in array])

    def l1(self, array):
        return np.array([np.sum(np.abs(i)) for i in array])

    def means_center(self, cluster_items: np.array):
        return cluster_items.mean(axis = 0)

    def median_center(self, cluster_items: np.array):
        return np.median(cluster_items,axis=0)

    def get_random_k(self, data):
        res = []
        for i in range(self.k):
            r = random.randint(0,len(data)-1)
            res.append(data[r])
            data = np.delete(data,r, 0)
        return np.array(res)

if __name__ == '__main__':
    data = genfromtxt('data/dataForClast.csv', delimiter=',',skip_header=1).astype(np.float)
    print('Data: ')
    print(data)
    clast = clustering(data, k=2, L_type= 2, means_or_median='median')
