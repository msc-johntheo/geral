
# coding: utf-8

# # Previsão de Séries Temporais com Programação Genética
# Este exercício tem o objetivo de utilizar Programação Genética para gerar um modelo de previsão de séries temporais. Os dados utilizados são referente ao número mensal de passageirosde linhas aéreas internacionais (em milhares/mês), desde janeir/1949 a dezembro/1960, conforme figura abaixo.
# ![grafico](serie-grafico.png)

# O primeiro passo foi criar o dataset com os dados da planilha fornecida. Separou-se em dois datasets: treinamento e teste.  
# Como **janela de periodicidade**, observou-se que a cada 12 passo o padrão se repetia, o que corresponde a um ano **(12 meses)**. Sendo assim utilizou-se esse valor para gerar uma matriz que contivesse 12 entradas e uma saída.

# In[1]:

import numpy as np
from numpy import genfromtxt

#lendo do arquivo
data = genfromtxt('serie.txt')
data = np.array([[data[index+pos] for pos in range(13)] for index in range(data.size - 12)])

#separando os conjuntos
DATA_TRAIN_SIZE = data[:,1].size - 12
data_train = data[:DATA_TRAIN_SIZE,:]
data_test = data[DATA_TRAIN_SIZE:,:]


# Para a implementação do algoritmo de Programação Genética, utilizou-se o **DEAP** *(Distributed Evolutionary Algorithms in Python)*. Abaixo realiza-se o import dos modulos que serão utilizados.

# In[2]:

import operator
import math
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# Define-se agora a estrutura da árvore que será gerada nas evoluções da PG. Esta árvore terá 12 entradas que correspnde a janela definida de 12 meses. Também serão definidas as funções e os terminais utilizados.

# In[3]:

#definicao da estrutura de um conjunto que terá 12 entradas
pset = gp.PrimitiveSet("MAIN", 12)

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)


# In[4]:

pset.addEphemeralConstant("rand101", lambda: random.randint(-180,180)) #terminal que gera um numero aleatorio entre [-1,1]


# Os argumentos(entradas) serão renomeados para facilitar o entendimento da arvore gerada posteriormente. Cada argumento representa um mês do ano.

# In[5]:

pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')
pset.renameArguments(ARG2='x3')
pset.renameArguments(ARG3='x4')
pset.renameArguments(ARG4='x5')
pset.renameArguments(ARG5='x6')
pset.renameArguments(ARG6='x7')
pset.renameArguments(ARG7='x8')
pset.renameArguments(ARG8='x9')
pset.renameArguments(ARG9='x10')
pset.renameArguments(ARG10='x11')
pset.renameArguments(ARG11='x12')


# Na sequencia, será definido o alicerce da PG:  
# - característica do problema(minimização ou maximização)  
# - método de geração da população(grow, full, half and half)  
# - função de fitness  
# - função de crossover  
# - função de mutação  
# - função de seleção  
# - definição da profundidade máxima da árvore

# In[6]:

#característica do problema
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#tipo do indivíduo = árvore
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#método de geração da população(árvore)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#compila a arvore gerada em uma função
toolbox.register("compile", gp.compile, pset=pset)

#função de fitness que compara a arvore gerada com os dados de treinamento
def fitness(individual):
    # Tranforma a expressao da arvore em um função invocável
    func = toolbox.compile(individual)
    
    # calcula o erro quadrado médio entre a expressao e o valor real 'y'
    sqerrors = ((func(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12)-y)**2 for x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y in data_train)
    
    return math.fsum(sqerrors) / len(data_train),

#define para o framework qual será a função de fitness utilizada
toolbox.register("evaluate", fitness)
#define o operador de seleção como torneio de tamanho 3
toolbox.register("select", tools.selTournament, tournsize=3)
#define função de cruzamento como corte em um ponto
toolbox.register("mate", gp.cxOnePoint)
#cria função de mutação utilizando o método de geração de árvore full
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
#define a função de mutação para o framework
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

#quando houver cruzamento o tamanho da arvore nao pode passar de 50
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))
#quando houver muatção o tamanho da arvore nao pode passar de 50
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))


# Com toda a estrutura definida agora é possível criar a função que de fato irá rodar  algoritmo.

# In[7]:

def algorithm():
    random.seed(318)
    #CXPB  - Probabilidade de crossover
    #MUTPB - Probabilidade de mutação
    #NGEN  - Numero de gerações
    CXPB, MUTPB, NGEN = 0.6, 0.1, 200

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Realiza a chamada do algortimo definido no capitulo 7 do KOZA.
    # Passa como parâmetros:
    #     - população
    #     - toolbox que é o objeto que representa o framework DEAP
    #     - parametros evolucionais (probabilidades de crossover e mutação e gerações)
    #     - objeto com os placeholders de estatística
    #     - halloffame - individuo(s) selecionado(s)
    #     - verbos que representa se será impresso nas saida a evolução do algoritmo
    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


# In[8]:

pop, log, hof = algorithm()


# ## Resultados

# ### 1. Análise de parâmetros e funções

# O individuo nessa implementação é gerado utilizando o método **HalfAndHalf** onde metade da expressao é gerada com o **grow** e outra metade com **full**.  
# Utilizou-se uma probalidade de crossover de 60% e de mutaçao de 10%. Como operador de crossover escolheu-se o cruzamento de ponto único ounde escolhe-se um ponto aleatório da árvore, corta-o e troca com o da outra árvore. Como operador de mutação, escolheu-se randomicamente um ponto em um individuo(árvore), então trocou-se a sub-arvore no ponto gerando um pedaço utilizando o método **full** onde cada folha tem profundidade máxima entre 0 e 2.  
# Para função de fitness utilizou-se o erro quadrado médio.  
# 
# Foram feitos varios testes(cenários) com combinações de funções para a Programação Genética, que podem ser resumidos nos três abaixo:
# - Cenário simplista
#     - Funções: soma e subtração.  
#     - Terminais: randomico entre [-1,1]  
#     - Obteve um erro médio de aproximadamente 9% e uma função bem simples. $f = x2+13$   
# 
# - Cenário coerente  
#     - Funções: soma, subtração, coseno, seno, negação, multiplicação.  
#     - Terminais: randomico entre [-180,180]  
#     - Obteve um erro médio de aproximadamente 5% e um função simples também.  $f = x1+27$
# 
# - Cenário complexo
#     - Funções: soma, subtração, divisão, coseno, seno, negação, multiplicação, exponencial
#     - Terminais: randomico entre [-180,180]  
#     - Obteve um erro médio superior a 10% e função muito complexa. Árvore muito grande.
#     
# Sendo assim o cenário coerente foi utilizado para apresentar os demais resultados.

# ### 2. Melhor expressão-S obtida

# In[9]:

str(hof[0])


# ### 3. Equação correspondente(simplificada)

# $$f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12) = x1 + 27$$

# ### 4. Planilha com os dados obtidos

# In[21]:

import pandas as pd

#Colunas da tabela
cols = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','VALOR REAL','VALOR PG','ERRO(%)']

#Compila  melhor indivíduo chamar a funçã com os resultados
function = gp.compile(hof[0],pset)
table_train = []
for x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y in data_train:
    pg = function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12)
    table_train.append([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y,pg,round(abs(100-pg*100/y),2)])

table_test = []    
for x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y in data_test:
    pg = function(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12)
    table_test.append([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y,pg,round(abs(100-pg*100/y),2)])

print('\n========\n|TREINO|\n========')
table_train = np.array(table_train)
df_train = pd.DataFrame(table_train,columns = cols)
print(df_train.to_string())

print('\n=======\n|TESTE|\n=======')
table_test = np.array(table_test)
df_test = pd.DataFrame(table_test,columns = cols)
print(df_test.to_string())


# ### 5. Gráfico com os dados reais e os gerados pela função obtida por PG

# In[11]:

import matplotlib.pyplot as plt

meses = np.arange(132)


fig, ax1 = plt.subplots()
ax1.set_xlabel("Meses")
ax1.set_ylabel("Passageiros")

line1 = ax1.plot(meses, np.concatenate((table_train[:,12],table_test[:,12]),axis=0), "b-", label="REAL")
line2 = ax1.plot(meses, np.concatenate((table_train[:,13],table_test[:,13]),axis=0), "r-", label="OBTIDO")


lns = line2 + line1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=4)

fig.set_size_inches(15, 5, forward=True)

plt.show()


# In[17]:

error_final = np.concatenate((table_train[:,14],table_test[:,14]),axis=0)
print('Erro médio treino: {}%'.format(sum(table_train[:,14])/len(table_train)))
print('Erro médio teste: {}%'.format(sum(table_test[:,14])/len(table_test)))
print('Erro médio total: {}%'.format(sum(error_final)/len(error_final)))


# ### 6. Considerações finais

# - Aumentando-se o numero de gerações não contribui para uma melhora do erro.  
# - As equações que envolviam seno e coseno não apresentaram-se melhor.  
# - Os dados iniciais apresentam um erro maior pois, devido a estrutura do problema, eles nao têm dados suficientes de treinamento.  
# - A adição e funções mais complexas como exponencial implica em incluir restrições para não dar erro ao calcular o fitness ou gerar um filho.
