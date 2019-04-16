#import libs
import pandas as pd
import numpy as np
import Gaussiano as gauss
import KnnMethod as knn
import Sum_Rule as sumRule
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing

#load datas
data1 = pd.read_csv('dataset/mfeat-fac', delimiter='\\s+', header=None)
data2 = pd.read_csv('dataset/mfeat-fou', delimiter='\\s+', header=None)
data3 = pd.read_csv('dataset/mfeat-kar', delimiter='\\s+', header=None)

x1 = data1.iloc[:,:].values
x2 = data2.iloc[:,:].values
x3 = data3.iloc[:,:].values

#normalize x1
scaler = preprocessing.StandardScaler().fit(x1)
x1Normalized = scaler.transform(x1)

# set y
y = []

#put labels
for i in range(len(x1)):
    if (i>=0 and i<=199):
        y.append(0)
    elif (i>=200 and i<=399):
        y.append(1)
    elif (i>=400 and i<=599):
        y.append(2)
    elif (i>=600 and i<=799):
        y.append(3)
    elif (i>=800 and i<=999):
        y.append(4)
    elif (i>=1000 and i<=1199):
        y.append(5)
    elif (i>=1200 and i<=1399):
        y.append(6)
    elif (i>=1400 and i<=1599):
        y.append(7)
    elif (i>=1600 and i<=1799):
        y.append(8)
    elif (i>=1800 and i<=1999):
        y.append(9)


def execute_kfold():
    label = np.array(y)

    accuracyKnn = []
    accuracyGauss = []

    kf = KFold(n_splits=10, random_state=None, shuffle=True)

    for train_index, test_index in kf.split(x1):
        # view1
        x1_train, x1_test = x1Normalized[train_index], x1Normalized[test_index]
        y1_train, y1_test = label[train_index], label[test_index]
        # view2
        x2_train, x2_test = x2[train_index], x2[test_index]
        y2_train, y2_test = label[train_index], label[test_index]
        # view3
        x3_train, x3_test = x3[train_index], x3[test_index]
        y3_train, y3_test = label[train_index], label[test_index]

        # gausiano runs
        # classification by view1
        returnGaussView1 = gauss.test_bayes(x1_train, y1_train, x1_test)
        pGauss1 = returnGaussView1[0]
        apriori = returnGaussView1[1]
        # classification by view2
        returnGaussView2 = gauss.test_bayes(x2_train, y2_train, x2_test)
        pGauss2 = returnGaussView2[0]
        # classification by view3
        returnGaussView3 = gauss.test_bayes(x3_train, y3_train, x3_test)
        pGauss3 = returnGaussView3[0]
        # sum rule
        accuracyGauss.append(sumRule.sum_rule(pGauss1, pGauss2, pGauss3, apriori, y1_test))

        # knn runs
        retunrKnnView1 = knn.knn_method(x1_train, y1_train, x1_test, 3)
        retunrKnnView2 = knn.knn_method(x2_train, y2_train, x2_test, 5)
        retunrKnnView3 = knn.knn_method(x3_train, y3_train, x3_test, 5)
        # sum rule
        accuracyKnn.append(sumRule.sum_rule(retunrKnnView1, retunrKnnView2, retunrKnnView3, apriori, y1_test))

    mean1 = np.mean(accuracyGauss)
    mean2 = np.mean(accuracyKnn)

    return round(mean1,3), round(mean2,3)

#ten kfold
run = 1
from tqdm import tqdm

globalAccuracy = []

for j in tqdm(range(run)):
    globalAccuracy.append(execute_kfold())

print(globalAccuracy)