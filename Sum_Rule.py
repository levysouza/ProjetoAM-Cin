import numpy as np
from sklearn.metrics import accuracy_score

def sum_rule(prob1, prob2, prob3, apriori, ytest):
      
    ypredict = []
     
    for i in range(len(prob1)):
        
        temp1 = np.add(prob1[i],prob2[i])
        
        temp2 = np.add(temp1,prob3[i])
        
        temp3 = np.add(temp2, apriori *(1-3)) # 1 - l, where l = 3
        
        ypredict.append(temp2.argmax()) 
    
    accuracy = accuracy_score(ytest, ypredict)
    
    return accuracy