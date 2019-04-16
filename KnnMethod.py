#imports
import pandas as pd
import numpy as np
from scipy.spatial import distance
from heapq import nsmallest

def knn_method(xtrain, ytrain, xtest, k):
    
    posteriori = []
    
    for i in range(len(xtest)):
    
        amostra = xtest[i]
    
        distancias = []
    
        for j in range(len(xtrain)):
        
            d = distance.euclidean(amostra, xtrain[j])
        
            classe =  ytrain[j]
        
            distancias.append((d, classe))
        
        kVizinhos = nsmallest(k, distancias)
        
        #get posteriori
        aux = calc_posteriori(kVizinhos,k)
        posteriori.append(aux)
    
    return posteriori

    
def calc_posteriori(kVizinhos,k):
    
    posteriori = []
    
    for i in range(10):
    
        total = 0
    
        for j in range(k):
        
            total = total + kVizinhos[j].count(i)
        
        posteriori.append(total/k)

    return posteriori