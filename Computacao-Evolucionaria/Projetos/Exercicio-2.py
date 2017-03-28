import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# INIT
toolbox = base.Toolbox()

# Natureza do problema
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Definindo a estrutura do indivíduo
IND_SIZE = 168  # Tamanho do individuo - 10 para cada gene
GENES = 12
TAM_GENE = 14
creator.create("Individual", list, fitness=creator.FitnessMax)

# funcao para gerar o gene com alelos 0 ou 1 randomicamente uniforme
toolbox.register("attr_bool", random.randint, 0, 1)

# funcao para gerar o indivíduo (nome, forma de gerar, Estrutura, funcao geradora, tamanho)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)

# funcao para gerar a populacao
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# funcao para retornar o individuo convertido para base inteira
def convert_individual_integer(individual):
    values = []
    for x in range(12):
        start = x * 14
        end = start + 14
        values.append(int("".join(str(i) for i in individual[start:end]), 2))
    # p1d,p1c,p1t,p2d....p4c,p4t
    return values


# funcao de fitness
def evaluate(individual):
    ind = convert_individual_integer(individual)
    coef = 5
    l1, l2, l3, l4 = 310, 380, 350, 285
    sum_c1 = ind[0] + ind[1] + ind[2]
    sum_c2 = ind[3] + ind[4] + ind[5]
    sum_c3 = ind[6] + ind[7] + ind[8]
    sum_c4 = ind[9] + ind[10] + ind[11]

    # Restricoes de peso nos compartimentos
    r1 = max(0, ind[0] + ind[3] + ind[6] + ind[9] - 10)
    r2 = max(0, ind[1] + ind[4] + ind[7] + ind[10] - 16)
    r3 = max(0, ind[2] + ind[5] + ind[8] + ind[11] - 8)

    # Restricoes de volume
    r4 = max(0.0, ind[0] * 0.48 + ind[3] * 0.65 + ind[6] * 0.58 + ind[9] * 0.39 - 6.8)
    r5 = max(0.0, ind[1] * 0.48 + ind[4] * 0.65 + ind[7] * 0.58 + ind[10] * 0.39 - 8.7)
    r6 = max(0.0, ind[2] * 0.48 + ind[5] * 0.65 + ind[8] * 0.58 + ind[11] * 0.39 - 5.3)

    # Restricoes de carga maxima
    r7 = max(0, ind[0] + ind[1] + ind[2] - 18)
    r8 = max(0, ind[3] + ind[4] + ind[5] - 15)
    r9 = max(0, ind[6] + ind[7] + ind[8] - 23)
    r10 = max(0, ind[9] + ind[10] + ind[11] - 12)

    # Restricoes de equilibrio
    r11 = abs((ind[0] + ind[3] + ind[6] + ind[9]) / 10 - (ind[1] + ind[4] + ind[7] + ind[10]) / 16)
    r12 = abs((ind[0] + ind[3] + ind[6] + ind[9]) / 10 - (ind[2] + ind[5] + ind[8] + ind[11]) / 8)
    r13 = abs((ind[2] + ind[5] + ind[8] + ind[11]) / 8 - (ind[1] + ind[4] + ind[7] + ind[10]) / 16)

    g = sum_c1 * l1 + sum_c2 * l2 + sum_c3 * l3 + sum_c4 * l4
    h = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13

    return g - coef * h,


# registra funcao de fitness
toolbox.register("evaluate", evaluate)

# registra crossOver
toolbox.register("mate", tools.cxTwoPoint)

# registra mutacao com probabilidade default de mudar cada gene de 5%
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# registra o metodo de selecao como torneio de tamanho 3
toolbox.register("select", tools.selTournament, tournsize=3)


# Imprimir indivíduo
def print_ind(individual):
    ind = convert_individual_integer(individual)
    print('  ' + '\t' + 'D' + '\t' + 'C' + '\t' + 'T')
    print('----------------------------')
    print('C1' + '\t|' + str(ind[0]) + '\t|' + str(ind[1]) + '\t|' + str(ind[2]))
    print('C2' + '\t|' + str(ind[3]) + '\t|' + str(ind[4]) + '\t|' + str(ind[5]))
    print('C3' + '\t|' + str(ind[6]) + '\t|' + str(ind[7]) + '\t|' + str(ind[8]))
    print('C4' + '\t|' + str(ind[9]) + '\t|' + str(ind[10]) + '\t|' + str(ind[11]))


    # Plotar Gráfico
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
    for tl in ax2.get_yticklabels():
        tl.set_color("g")

    ax3 = ax1.twinx()
    line3 = ax3.plot(gen, max, "y-", label="Maximum Fitness")
    ax3.set_ylabel("Size")
    for tl in ax3.get_yticklabels():
        tl.set_color("y")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()


def main():
    random.seed(94)

    # cria populacao inicial
    pop = toolbox.population(n=50)

    # CXPB - probabilidade de crossover
    # MUTPB - probabilidade de mutacao
    # NGEN - numero de geracoes
    CXPB, MUTPB, NGEN = 0.8, 0.02, 200

    # stats a serem guardados
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)

    # Roda o algoritmo
    pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats)

    # Seleciona o melhor individuo da populacao resultante
    best_ind = tools.selSPEA2(pop, 1)

    # Imprime as infromações do melhor individuo
    print_ind(best_ind[0])

    # Plota o Gráfico
    plot_log(logbook)


if __name__ == "__main__":
    main()
