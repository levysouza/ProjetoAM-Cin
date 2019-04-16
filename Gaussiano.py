import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from numpy.linalg import inv, det


def parameter_estimation(xtrain,ytrain):
    
    bayes = GaussianNB()
    bayes.fit(xtrain,ytrain)

    apriori = bayes.class_prior_
    mean = bayes.theta_
    std = bayes.sigma_
    totalClass = apriori.shape[0]

    return apriori, mean, std, totalClass


def density_calculation(x,parameter):
    
    density = []
    
    amostra = x
    
    apriori = parameter[0]
    mean = parameter[1]
    std = parameter[2]
    totalClass = parameter[3]

    for j in range(totalClass):
        
        diferenca = amostra - mean[j]
        
        inversa = inv(np.identity(amostra.shape[0]) * std[j])
        
        determinante = det(inversa)
        
        transp = diferenca.reshape((1, amostra.shape[0]))
        
        calc1 = np.power(2*np.pi,-(amostra.shape[0]/2)) 
    
        calc2 = np.power(determinante,1/2)

        temp1 = np.dot(diferenca.reshape(1, amostra.shape[0]),inversa)
        
        temp2 = np.exp( - 1/2 * (np.dot(temp1,diferenca.reshape(amostra.shape[0],1))))
    
        densidade = float(calc1 * calc2 * temp2)
    
        #density by a priori
        density.append(densidade * apriori[j])
    
    return density


def posteriori_calculation(density):
    
    posteriori = []
    
    evidence = sum(density)
    
    for i in range(len(density)):
        
        temp = density[i] / evidence
        
        posteriori.append(temp)
    
    return posteriori



def test_bayes(xtrain, ytrain, xtest):
    
    posteriori = []
    
    parameter = parameter_estimation(xtrain,ytrain)
    
    for i in range(len(xtest)):
        
        amostra = xtest[i]
        
        density = density_calculation(amostra,parameter)
        
        posteriori.append(posteriori_calculation(density))
    
    return posteriori, parameter[0]
