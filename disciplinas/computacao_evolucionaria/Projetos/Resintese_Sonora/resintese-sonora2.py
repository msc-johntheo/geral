#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

#Util
import operator
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Genetic Programming
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Audio
import librosa
import librosa.display
import thinkdsp


# DADOS INICIAIS
wave = thinkdsp.read_wave('sounds/92002__jcveliz__violin-origional.wav')
target = wave.segment(1.18995, 0.62)


# FUNCOES AUXILIARES PARA FITNESS
def getMFCC(y, sr, is_print=False):
    ### MFC ###
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)

    if is_print:
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

    if is_print:
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


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, target):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x

    gen_ys = np.array([func(x) for x in range(len(target.ys))])
    gen_sr = target.framerate

    gen_mfcc = getMFCC(gen_ys, gen_sr, False)
    target_mfcc = getMFCC(target.ys, target.framerate, False)

    return mfcc_squared_error(gen_mfcc,target_mfcc),


toolbox.register("evaluate", evalSymbReg, target=target)
# toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    # CXPB  - Probabilidade de crossover
    # MUTPB - Probabilidade de mutação
    # NGEN  - Numero de gerações
    CXPB, MUTPB, NGEN = 0.6, 0.1, 50

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=mstats,
                                   halloffame=hof, verbose=True)
    # pop, log = gp.harm(pop, toolbox, 0.5, 0.1, 40, alpha=0.05, beta=10, gamma=0.25, rho=0.9, stats=mstats,
    #                   halloffame=hof, verbose=True)
    # print log
    print(str(hof[0]))
    return pop, log, hof


if __name__ == "__main__":
    pop, log, hof = main()
