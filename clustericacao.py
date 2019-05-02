from Util import readDatabase, getInitialG, getInitialL, getInitialU, argmaxIndex, saveParameters, saveMatrix, saveResults
import numpy
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from equacoes import J, update_G, update_L, update_U

#definição dos datasets (matrizes de dissimilaridade dos datasets com dados já normalizados(z))
#E1 = fac, E2 = fou, E3 = kar
E1, E2, E3 = readDatabase()

G_lista_final = []
L_lista_final = []
U_lista_final = []
J_lista_final = []
todos_J_exec = []

for exe in range(100):
    #definição dos parâmetros
    print("Inicializando parametros iniciais da execucao "+str(exe+1))
    K = 10
    m = 1.6
    T = 150
    epsilon = 10 ** -10
    E = numpy.array([E1, E2, E3])
    n = len(E[0])
    p = len(E)
    G = getInitialG(E,K) #protótipo do grupo - vetor de medoids (10 x 3)
    L = getInitialL(E,K) #lambda - vetor de relevancia (pesos) (10 x 3) -- inicialmente todos 1
    U = getInitialU(m,E,K,G,L) #matriz de graus de pertinencia de cada exemplo
    j_t = J(m, E, K, G, L, U, n, p) #calculo de primeira funcao objetivo

    #guardar todos os valores de J, G, L e U
    j_check = []
    j_check.append(j_t)
    g_check = []
    g_check.append(G)
    l_check = []
    l_check.append(L)
    u_check = []
    u_check.append(U)

    for t in range(T):
        G = update_G(m, E, K, G, L, U, n, p)
        L = update_L(m, E, K, G, L, U, n, p)
        U = update_U(m, E, K, G, L, U, n, p)

        j_ant = j_t  # armazena temporariamente o valor anterior de j
        j_t = J(m,E,K,G,L,U, n, p)
        j_diferenca = j_t-j_ant

        j_check.append(j_t)
        g_check.append(G)
        l_check.append(L)
        u_check.append(U)

        if(j_diferenca<epsilon):
            print("break por diferenca menor de que epsilon")
            break

    todos_J_exec.append(j_check)
    G_lista_final.append(g_check[-1])
    L_lista_final.append(l_check[-1])
    U_lista_final.append(u_check[-1])
    J_lista_final.append(j_check[-1])

#print(G_lista_final)
#print(L_lista_final)
#print(U_lista_final)
#print(J_lista_final)
#print(j_check)

melhorExe = argmaxIndex(J_lista_final, [])
print("A melhor execução foi a execução "+str(melhorExe+1))

#salvar parâmetros
J_lista_all = numpy.array(todos_J_exec)
J_lista_final = numpy.array(J_lista_final)
saveParameters(U_lista_final, G_lista_final, L_lista_final, J_lista_all, J_lista_final)

#formar base CRISP
matrix_fuzzy = U_lista_final[melhorExe]
linha = 0
controle_classe = 0
valor_classe = 0
clusters = []
classes = []

for i in matrix_fuzzy:
    ind_cluster = numpy.argmax(i)
    clusters.append(ind_cluster)
    if (controle_classe == 199):
        classes.append(valor_classe)
        valor_classe += 1
        controle_classe = 0
    else:
        classes.append(valor_classe)
        controle_classe += 1
    i = i.tolist()
    for j in range(len(i)):
        if (ind_cluster == j):
            matrix_fuzzy[linha, j] = int(1)
        else:
            matrix_fuzzy[linha, j] = int(0)
    linha += 1

#salvar matrizes crisp, y
matrix_crisp = matrix_fuzzy.astype('int')
#matrix_crisp = matrix_crisp.tolist()
saveMatrix(matrix_crisp, classes, clusters)

value_index_rand = adjusted_rand_score(classes, clusters)
saveResults(matrix_crisp, value_index_rand, melhorExe)

#df = pd.DataFrame({'clusters' : clusters, 'labels' : classes})
#teste=df.groupby('clusters').apply(lambda cluster: cluster.sum()/cluster.count())
#print(teste)

#resultados
#lambda: vetor de medoids
#crisp
#num de obj de cada grupo
#indexRand: indice Rand corrigido
