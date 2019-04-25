from Util import readDatabase, getInitialG, getInitialL, getInitialU, argmaxIndex
import numpy
from equacoes import J, update_G, update_L, update_U

#definição dos datasets (matrizes de dissimilaridade dos datasets com dados já normalizados(z))
#E1 = fac, E2 = fou, E3 = kar
E1, E2, E3 = readDatabase()

G_lista_final = []
L_lista_final = []
U_lista_final = []
J_lista_final = []

for exe in range(4):
    #definição dos parâmetros
    print("Inicializando parametros iniciais da execucao "+str(exe))
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
    G_lista_final.append(g_check[-1])
    L_lista_final.append(l_check[-1])
    U_lista_final.append(u_check[-1])
    J_lista_final.append(j_check[-1])

#print(G_lista_final)
#print(L_lista_final)
#print(U_lista_final)
#print(J_lista_final)

melhorExe = argmaxIndex(J_lista_final,[])
print("A melhor execução foi a execução "+str(melhorExe))


#resultados
#lambda: vetor de medoids
#crisp
#num de obg de cada grupo
#indexRand: indice Rand corrigido
