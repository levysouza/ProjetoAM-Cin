import numpy
from sklearn import preprocessing
import scipy
import random

def gerarLabel():
    y = []
    for i in range(2000):
        if (i >= 0 and i <= 199):
            y.append(0)
        elif (i >= 200 and i <= 399):
            y.append(1)
        elif (i >= 400 and i <= 599):
            y.append(2)
        elif (i >= 600 and i <= 799):
            y.append(3)
        elif (i >= 800 and i <= 999):
            y.append(4)
        elif (i >= 1000 and i <= 1199):
            y.append(5)
        elif (i >= 1200 and i <= 1399):
            y.append(6)
        elif (i >= 1400 and i <= 1599):
            y.append(7)
        elif (i >= 1600 and i <= 1799):
            y.append(8)
        elif (i >= 1800 and i <= 1999):
            y.append(9)
    return y

def normalize_columns(data):
    rows, cols = data.shape
    for col in range(0, cols):
        minimo = data[:, col].min()
        maximo = data[:, col].max()

        if (minimo != maximo):
            denominador = maximo - minimo
            normazu = (data[:, col] - minimo) / denominador
            data[:, col] = normazu  # [max,min] -> [0,1]
            # data[:,col] = (normazu*2) - 1 # [max,min] -> [-1,1]
        else:
            data[:, col] = 0  # column 'col' - numpyarray notation

def readDatabase():
    #Leitura dos dados
    print("Iniciando leitura dos dados")
    src = "D:/Google Drive/Doutorado/AM/Projeto/Dataset/"
    fac = numpy.genfromtxt(src + "mfeat-fac.txt")
    fou = numpy.genfromtxt(src + "mfeat-fou.txt")
    kar = numpy.genfromtxt(src + "mfeat-kar.txt")

    #Gerar labels (0-9)
    y = gerarLabel()
    print("Leitura dos dados finalizada")

    #Normalizar dados z = (x - u) / s
    print("Iniciando normalização dos dados")
    scaler = preprocessing.StandardScaler().fit(fac)
    facNormalized = scaler.transform(fac)

    scaler = preprocessing.StandardScaler().fit(fou)
    fouNormalized = scaler.transform(fou)

    scaler = preprocessing.StandardScaler().fit(kar)
    karNormalized = scaler.transform(kar)
    print("Normalização dos dados finalizada")

    #Calculando matrizes de dissimilaridade
    print("Iniciando calculo de matrizes de dissimilaridade")
    print("--fac ")
    facDM = dissimilarityMatrix(facNormalized)
    print("--fou ")
    fouDM = dissimilarityMatrix(fouNormalized)
    print("--kar ")
    karDM = dissimilarityMatrix(karNormalized)
    print("Calculo finalizado de matrizes de dissimilaridade")

    return facDM, fouDM, karDM

def dissimilarityMatrix(dataset): # retorna a D(nxn) do dataset de n instancias:
    dist = scipy.spatial.distance.euclidean
    mat = []
    for i in dataset:
        row = []
        for j in dataset:
            row += [dist(i,j)]
        mat.append(row)
    mat = numpy.array(mat)
    return mat

def getInitialG(E,K):
    G=[]
    p = len(E)
    for k in range(K):
        g_k = []
        for j in range(p):
            random_temp = random.randint(0,len(E[j])-1)
            g_k.append(random_temp)
        G.append(g_k)
    return numpy.array(G)

def getInitialL(E,K):
    L = []
    p = len(E)
    for k in range(K):
        l_k = []
        for h in range(p):
            l_k.append(1.)
        L.append(l_k)
    return numpy.array(L)

def getInitialU(m,E,K,G,L): #equacao 6
    U = []
    n = len(E[0])
    p = len(E)
    for i in range(n):
        u_i = []
        for k in range(K):
            summ = 0
            for h in range(K):
                numerator = 0
                denominator = 0
                for j in range(p):
                    #numerator += L[k][j] * dist(E[i],G[k][j])
                    numerator += L[k][j] * E[j][i][G[k][j]]
                for j in range(p):
                    #denominator+= L[h][j] * dist(E[i],G[h][j])
                    denominator+= L[h][j] * E[j][i][G[h][j]]
                summ += ( numerator /(denominator + 1e-25 )) ** (1./(m-1.))
            u_i_k = (( summ  ) + 1e-25  ) ** -1.
            u_i.append(u_i_k)
        U.append(u_i)
    return numpy.array(U)

def argminIndex(list,exclude):
    argmin = min(list)
    argmax = max(list)
    indexList = []
    for i in range(len(list)):
        if (list[i] == argmin) :
            if i in exclude:
                list[i] = argmax
                i = argminIndex(list,exclude)
            indexList.append(i)
    return indexList[0]

def argmaxIndex(list,exclude):
    argmin = min(list)
    argmax = max(list)
    indexList = []
    for i in range(len(list)):
        if (list[i] == argmin) :
            if i in exclude:
                list[i] = argmax
                i = argminIndex(list,exclude)
            indexList.append(i)
    return indexList[0]