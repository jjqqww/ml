# -*- coding: utf-8 -*-
"""
Spyder Editor

kmeans clustering implementation
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def init_centroids(data,k):
    n, dim = data.shape
    res = np.zeros((k, dim))
    for i in xrange(k):
        idx = int(np.random.uniform(0, n))
        res[i,:]= data[idx,:]
    print res
    return res

def kmeans(data, k):
    samples, dim = data.shape
    #print samples
    #cluster_record = np.array([[0,np.finfo(np.float64).max]]*samples) #index, value
    cluster_record = np.zeros([samples,2]) #index, value
    #print type(cluster_record) #numpy.ndaddray
    keep_running = True
    centroids = init_centroids(data,k)

    max_iter = 100000
    iter = 0
    while keep_running and iter< max_iter:
        keep_running = False

        for i in xrange(samples):
            min_distance = np.finfo(np.float64).max
            min_index = 0
            for j in xrange(k):
                c= centroids[j]
                x = data[i]
                distance = np.sqrt(np.sum(np.power((x-c),2)))
                min_index = j if distance<min_distance else min_index
                min_distance = distance if distance<min_distance else min_distance
                
                #print min_index
            if int(min_index) != int(cluster_record[i,0]):
                keep_running = True
                cluster_record[i:] = min_index, min_distance
                #print cluster_record
        for i in xrange(k):
            points = data[cluster_record[:,0]==i]
            centroids[i,:] = np.mean(points, axis=0)
        iter+=1
        plot(data, centroids, cluster_record)
    print "done"
    return centroids, cluster_record

def plot(data, centroids, cluster_record):
    
    #print cluster_record
    n, dim = data.shape
    
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if centroids.shape[0] > len(mark):
        raise Exception("k is too large")
    for i in xrange(n):
        my_mark = mark[int(cluster_record[i][0])]
        plt.plot(data[i,0], data[i,1], my_mark)
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in xrange(centroids.shape[0]):
        plt.plot(centroids[i,0], centroids[i,1], mark[i])
    plt.show()

def main():
    data = []
    f = r"C:\Users\jin\Documents\python_machine_learning\data.txt"
    with open(f) as file_in:
        for line in file_in.readlines():
            tmp = line.strip().split(" ")
            data.append([float(tmp[0]), float(tmp[1])])
    data = np.array(data)
    #print type(data)
    #print data.shape
    for i in xrange(data.shape[0]):
        plt.plot(data[i,0], data[i,1], 'or')
    plt.show()
    centroids, cluster_record = kmeans(data, 3)
    plot(data, centroids, cluster_record)
    


if __name__ == '__main__':
    main()