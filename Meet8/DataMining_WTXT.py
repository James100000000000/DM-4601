from ast import Str
from cProfile import label
from code import interact
from email import iterators
from hashlib import new
from re import I, T
from tkinter.messagebox import showwarning
from turtle import shape
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time
import itertools

datasetPath = "D:\dataku.txt"
dataset = np.loadtxt(datasetPath,delimiter="")
k = 2 #jml cluster
iterationCounter = 0 #
input = dataset

def initCCentroid(dataIn, k):
    result = dataIn[np.random.choice(dataIn.shape[0],k,replace=False)]
    return result
def plotClusterResult(listClusterMembers,centroid,iteraton,converged):
    n = listClusterMembers.__len__()
    color = iter(cm.rainbow(np.linspace(0,1,n)))
    plt.figure("result")
    plt.clf()
    plt.title("Iteration-"+iteraton)
    marker = itertools.cycle(('.','*','^','x','+'))
    for i in range(n):
        col = next(color)
        memberCluster = np.asmatrix(listClusterMembers[i])
        plt.scatter(np.ravel(memberCluster[:,0]), np.ravel(memberCluster[:,1]), 
                    marker=marker.__next__(),s=100,c=col, label="klaster"+str(i+1))
    for i in range(n):
        plt.scatter((centroid[i,0]),(centroid[i,1]),marker=marker.__next__(), e=col, label="centroid-"+Str(i+1) )
    if(converged == 0):
        plt.legend()
        plt.ion()
        plt.show()
        plt.pause(0,1)
    if(converged == 1):
        plt.legend()
        plt.show(vlock=True)
def kMeans(data,centroidInit):
    ncluster = k
    global iterationCounter
    centroidInit = np.matrix(centroidInit)
    while(True):
        iterationCounter +=1
        euclideanMatrixAllCluster = np.ndarray(shape=(data.shape[0],0))
        for i in range(0,ncluster):
            centroidRepeated = np.repeat(centroidInit[i,:],data.shape[0],axis=0)
            deltaMatrix = abs(np.subtract(data,centroidRepeated))
            euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
            euclideanMatrixAllCluster = \
                np.concatenate((euclideanMatrixAllCluster,euclideanMatrix),axis=1)
            clusterMatrix = np.ravel(np.argmin(np.matrix(euclideanMatrixAllCluster),axis=1))
            listClusterMember=[[]for i in range(k)]
        for i in range(0, data.shape[0]):
            listClusterMember[np.ascalar(clusterMatrix[i])].append(data[i,:])

        newCentoid = np.ndarray(shape=(0,centroidInit.shape[1]))
        for i in range(0,ncluster):
            memberCluster = np.asmatrix(listClusterMember[i])
            centroidCluster = memberCluster.mean(axis=0)
            newCentoid = np.concatenate((newCentoid, centroidCluster),axis=0)

        print("iter: ", iterationCounter)
        print("centroid: ", newCentoid)
        if((centroidInit == newCentoid).all()):
            break
        centroidInit = newCentoid
        plotClusterResult(listClusterMember,centroidInit, str(iterationCounter),0)
        time.sleep(1)
    return listClusterMember, centroidInit 
centroidInit = initCCentroid(input, k)
clusterResult, centroid = kMeans(input,centroidInit)
plotClusterResult(clusterResult,centroid, str(iterationCounter)+"(converged)",1)
