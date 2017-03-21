#   Exercicio 1 - Garrafas
#   Os passos básicos para utilizar o DEAP são:
#       1 - Definir a natureza do problema(max/min single/multi objetivo)
#       2 - Definir a estrutura(lista, set ou estrutura particular) do indivídudo da populção e como ele é
#       avaliado(relacionamento com fitness). Logo após isso deve-se registrar a funcao de como será gerado o indivíduo
#       3 - Registrar a função que irá gerar a população com babse na funcao do individuo
#       4 - Definir a função de fitness. Essa função sempre retorna tuplas com valores para cada objetivo.
#       Casos de objetivo simples retornam uma tupla com segundo parametro nulo
#       5 - Definir quais operadores usar(evaluate, mate, mutate, select)
#       6 - Algoritmo main rodando todos os passos. Pode-se fazer manualmente ou utilizar algo da toolbox

import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# INIT
toolbox = base.Toolbox()

# ----------
# NATUREZA DO PROBLEMA
# ----------
# Definindo a natureza do problema. No caso um problema de maximização
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Definindo a estrutura do indivíduo
IND_SIZE = 20  # Tamanho do individuo - 10 para cada gene
GENES = 2
creator.create("Individual", list, fitness=creator.FitnessMax)

# funcao para gerar o gene com valores 0 ou 1 randomicamente uniforme
toolbox.register("attr_bool", random.randint, 0, 1)
# funcao para gerar o indivíduo (nome, forma de gerar, Estrutura, funcao geradora, tamanho)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)

# funcao para gerar a populacao
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# funcao de fitness
def evaluate(individual):
    l = int("".join(str(i) for i in individual[:int(IND_SIZE/GENES)]), 2)    # gene quantidade de garrafas de leite
    s = int("".join(str(i) for i in individual[int(IND_SIZE/GENES):]), 2)    # gene quantidade de garrafas de suco
    g = (5*l + 4.5*s)/7375                                                   # funcao objetivo normalizada [0,1]
    h1 = max(0, (0.06*l + 0.05*s - 60)/15)                                   # funcao de restricao 1 normalizada [0,1]
    h2 = max(0, (10*l + 20*s - 15000)/3750)                                  # funcao de restricao 2 normalizada [0,1]
    h3 = max(0, (l - 800)/200)                                               # funcao de restricao 3 normalizada [0,1]
    h4 = max(0, (s - 750)/187.5)                                             # funcao de restricao 4 normalizada [0,1]
    return g - (h1+h2+h3+h4),                                                # fitness normalizado [0,1]

# ----------
# OPERADORES
# ----------
# registra funcao de fitness
toolbox.register("evaluate", evaluate)

# registra crossOver
toolbox.register("mate", tools.cxTwoPoint)

# registra mutacao com probabilidade default de mudar cada gene de 5%
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# registra o metodo de selecao como torneio de tamanho 3
toolbox.register("select", tools.selTournament, tournsize=2)


# ----------
# ALGORITMO
# ----------
def main():
    random.seed(94)

    # cria populacao inicial
    pop = toolbox.population(n=100)

    # MU - numero de individuos selecionados para a prox geracao
    # LAMBDA - numero de filhos a serem gerados
    # CXPB - probabilidade de crossover
    # MUTPB - probabilidade de mutacao
    # NGEN - numero de geracoes
    MU, LAMBDA_, CXPB, MUTPB, NGEN = 5, 7, 0.5, 0.02, 100
    #CXPB, MUTPB, NGEN = 0.5, 0.2, 200  # LUCR0 = 5138

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)

    #Algoritmos do livro “Evolutionary Computation 1 : Basic Algorithms and Operators”
    #roda o algoritmo do capitulo 7
    pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, verbose=True)

    #roda o algoritmo (mi  ,lambda)
    #pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, MU, LAMBDA_, CXPB, MUTPB, NGEN, stats=stats, verbose=True)

    # roda o algoritmo (mi + lambda)
    #pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA_, CXPB, MUTPB, NGEN, stats=stats, verbose=True)

    #Seleciona o melhor individuo da populacao resultante
    best_ind = tools.selSPEA2(pop, 1)

    #Imprime as infromações do melhor individuo
    print_ind(best_ind[0])

    plot_log(logbook)

def print_ind(individual):
    l = int("".join(str(i) for i in individual[:int(IND_SIZE / GENES)]), 2)  # gene quantidade de garrafas de leite
    s = int("".join(str(i) for i in individual[int(IND_SIZE / GENES):]), 2)  # gene quantidade de garrafas de suco
    g = 5*l + 4.5*s
    print('Quantidade de garrafas de leite: ' + str(l))
    print('Quantidade de garrafas de suco: ' + str(s))
    print('Lucro ótimo: ' + str(g))


def plot_log(logbook):
    gen = logbook.select("gen")
    min = logbook.select("min")
    avg = logbook.select("avg")
    max = logbook.select("max")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, min, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, avg, "g-", label="Average Fitness")
    ax2.set_ylabel("Size", color="g")
    for tl in ax2.get_yticklabels():
        tl.set_color("g")

    ax3 = ax1.twinx()
    line3 = ax3.plot(gen, max, "y-", label="Maximum Fitness")
    ax3.set_ylabel("Size", color="y")
    for tl in ax3.get_yticklabels():
        tl.set_color("y")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()

if __name__ == "__main__":
    main()
