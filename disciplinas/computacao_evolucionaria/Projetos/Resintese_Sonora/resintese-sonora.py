from __future__ import print_function

# We'll need numpy for some mathematical operations
import numpy as np
from numpy import linspace, sin, pi

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as ms

# and IPython.display for audio output
import IPython.display

# Librosa for audio
import librosa
import thinkdsp
import thinkplot

# And the display module for visualization
import librosa.display

import operator
import math
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# DADOS INICIAIS
wave = thinkdsp.read_wave('sounds/92002__jcveliz__violin-origional.wav')
target = wave.segment(1.18995, 0.62)


def getMFCC(y, sr, isPrint):
    ### MFC ###
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)

    if (isPrint):
        # Make a new figure
        plt.figure(figsize=(12, 4))

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        # Put a descriptive title on the plot
        plt.title('mel power spectrogram')

        # draw a color bar
        plt.colorbar(format='%+02.0f dB')

        # Make the figure layout compact
        plt.tight_layout()

    ### MFCC ###

    # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    # Let's pad on the first and second deltas while we're at it
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    if (isPrint):
        # How do they look?  We'll show each in its own subplot
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        librosa.display.specshow(mfcc)
        plt.ylabel('MFCC')
        plt.colorbar()

        plt.subplot(3, 1, 2)
        librosa.display.specshow(delta_mfcc)
        plt.ylabel('MFCC-$\Delta$')
        plt.colorbar()

        plt.subplot(3, 1, 3)
        librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
        plt.ylabel('MFCC-$\Delta^2$')
        plt.colorbar()

        plt.tight_layout()

    # For future use, we'll stack these together into one matrix
    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    return M


def mfcc_squared_error(mfcc_target, mfcc_generated):
    erro_mfcc = mfcc_target - mfcc_generated
    squared_error_mfcc = erro_mfcc ** 2
    return np.sum(squared_error_mfcc)


# In[13]:

def generate_random_wave(complexity):
    f = np.random.randint(1000, size=complexity)
    a = np.random.randint(100, size=complexity)
    rand_wave = 0
    for i in range(complexity):
        sin_sig = thinkdsp.SinSignal(freq=random.randint(300, 2000) * np.random.randn() ** 2,
                                     amp=random.randint(100, 1000) * np.random.randn() ** 2, offset=0)
        rand_wave += sin_sig.make_wave(duration=0.5, start=0, framerate=44100)
    return rand_wave


def gen_rand_sin():
    freq = random.randint(300, 2000) * np.random.randn() ** 2
    amp = random.randint(100, 1000) * np.random.randn() ** 2
    return thinkdsp.Sinusoid(freq=freq, amp=amp, offset=0, func=np.sin)


def sum_wave(sinusoid1, sinusoid2):
    return sinusoid1 + sinusoid2




"""
========= GENETICAL PROGRAMMING ===============
"""
# definicao da estrutura de um conjunto que terá 1 entrada. A onda fundamental
pset = gp.PrimitiveSetTyped("MAIN", [thinkdsp.Sinusoid], thinkdsp.Sinusoid)
#pset.addPrimitive(thinkdsp.ComplexSinusoid,[thinkdsp.Signal],thinkdsp.Signal)
pset.addPrimitive(sum_wave, [thinkdsp.Sinusoid, thinkdsp.Sinusoid], thinkdsp.Sinusoid)
pset.addTerminal(thinkdsp.SinSignal(), thinkdsp.Sinusoid)
pset.addEphemeralConstant("rand_sin", lambda: gen_rand_sin(), thinkdsp.Sinusoid)

# característica do problema
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# tipo do indivíduo = árvore
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# método de geração da população(árvore)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# compila a arvore gerada em uma função
toolbox.register("compile", gp.compile, pset=pset)


# função de fitness que compara a arvore gerada com os dados de treinamento
def fitness(individual):
    # Tranforma a expressao da arvore em um função invocável
    func = toolbox.compile(individual)
    return 1,


# define para o framework qual será a função de fitness utilizada
toolbox.register("evaluate", fitness)
# define o operador de seleção como torneio de tamanho 3
toolbox.register("select", tools.selTournament, tournsize=3)
# define função de cruzamento como corte em um ponto
toolbox.register("mate", gp.cxOnePoint)
# cria função de mutação utilizando o método de geração de árvore full
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# define a função de mutação para o framework
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# quando houver cruzamento o tamanho da arvore nao pode passar de 50
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))
# quando houver muatção o tamanho da arvore nao pode passar de 50
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))


# In[59]:

def algorithm():
    random.seed(318)
    # CXPB  - Probabilidade de crossover
    # MUTPB - Probabilidade de mutação
    # NGEN  - Numero de gerações
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


def main():
    pop, log, hof = algorithm()


if __name__ == "__main__":
    main()
