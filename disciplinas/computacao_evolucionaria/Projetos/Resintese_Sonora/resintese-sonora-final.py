import operator
import math
import random
import numpy as np
from numpy import linspace, pi
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

# multiprocessing
import multiprocessing
import scoop
from scoop import futures


def extract_features(wave):
    # wave.normalize()
    X = np.array(wave.ys, dtype=np.float32)
    sample_rate = wave.framerate
    features = np.empty((0, 193))

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

    return features


# FITNESS 1
def mag_pha_s_error(target, generated):
    partitions = 11
    # obtendo os parametros
    target_spectrum = target.make_spectrum()
    target_pha = target_spectrum.angles
    target_mag = np.absolute(target_spectrum.hs)
    generated_spectrum = generated.make_spectrum()
    generated_pha = generated_spectrum.angles
    generated_mag = np.absolute(generated_spectrum.hs)

    # dividindo em bandas(partitions)
    target_pha_bin = np.array(np.array_split(target_pha, partitions, axis=0))
    generated_pha_bin = np.array(np.array_split(generated_pha, partitions, axis=0))
    target_mag_bin = np.array(np.array_split(target_mag, partitions, axis=0))
    generated_mag_bin = np.array(np.array_split(generated_mag, partitions, axis=0))

    pha_error = ((target_pha_bin - generated_pha_bin) ** 2).mean(axis=1)
    mag_error = ((target_mag_bin - generated_mag_bin) ** 2).mean(axis=1)

    pha_error_norm = pha_error / pha_error.max()
    mag_error_norm = mag_error / mag_error.max()

    pha_sme = pha_error_norm.mean()
    mag_sme = mag_error_norm.mean()

    return (pha_sme + mag_sme) / 2


# FITNESS 2
def mag_fre_s_error(target, generated):
    target_peaks = np.array(target.make_spectrum().peaks())
    generated_peaks = np.array(generated.make_spectrum().peaks())
    mag_sme, fre_sme = ((target_peaks - generated_peaks) ** 2).mean(axis=0)

    return mag_sme, fre_sme


# FITNESS 3
def features_diff(target, generated):
    diff = extract_features(target) - extract_features(generated)
    return ((diff / diff.max()) ** 2).mean(axis=0)


# Funcao para calculo de fitness
def evalSymbReg(individual, target):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    gen_ys = np.array([func(x, f) for x in t])
    gen_sr = target.framerate
    generated = thinkdsp.Wave(ys=gen_ys, framerate=gen_sr)

    # gen_mfcc = getMFCC(gen_ys, gen_sr, False)
    # target_mfcc = getMFCC(target.ys, target.framerate, False)
    # result = mfcc_squared_error(gen_mfcc,target_mfcc)
    # result = mfcc_squared_error(gen_ys,target.ys)
    # result = mag_pha_s_error(target,generated)
    result = features_diff(target, generated)
    return result,


def plot_log(logbook, title):
    """Funcao auxiliar para plotar gráfico"""
    gen = logbook.select("gen")
    min = logbook.chapters["fitness"].select("min")
    avg = logbook.chapters["fitness"].select("avg")
    max = logbook.chapters["fitness"].select("max")

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Fitness", color="b")

    line1 = ax1.plot(gen, min, "b-", label="Mínimo")
    line2 = ax1.plot(gen, avg, "g-", label="Médio")
    line3 = ax1.plot(gen, max, "y-", label="Máximo")

    lns = line3 + line2 + line1
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=4)

    fig.set_size_inches(15, 5, forward=True)

    plt.show()



# DADOS INCIAIS
wave = thinkdsp.read_wave('sounds/92002__jcveliz__violin-origional.wav')
# target = wave.segment(1.18995, 0.62)
target = wave.segment(1.18995, 0.2)
t = linspace(0, target.duration, target.duration * target.framerate)
f = target.make_spectrum().peaks()[1][1]

# GENETIC PROGRAMMING CONFIG
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(math.sin, 1)
pset.addTerminal(2 * pi)
if not scoop.IS_ORIGIN:
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
    pset.addEphemeralConstant("rand", lambda: random.randint(100, 1000))
pset.renameArguments(ARG0='t')
pset.renameArguments(ARG1='f')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg, target=target)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=50))

def fm_mod(amp_carrier, freq_carrier, amp_wave, freq_wave):
    return amp_wave*math.sin(freq_wave*t + amp_carrier*math.sin(freq_carrier*t))

def main():
    random.seed(random.randint(1, 100))

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)

    # CXPB  - Probabilidade de crossover
    # MUTPB - Probabilidade de mutação
    # NGEN  - Numero de gerações
    CXPB, MUTPB, NGEN = 0.75, 0.12, 20

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)
    # pop, log = algorithms.eaMuCommaLambda(population=pop,
    #                                      toolbox=toolbox, 
    #                                      mu=200, 
    #                                      lambda_=200, 
    #                                      cxpb=CXPB, 
    #                                      mutpb=MUTPB, 
    #                                      ngen=NGEN, 
    #                                      stats=mstats,
    #                                      halloffame=hof, 
    #                                      verbose=True)
    # pop, log = gp.harm(pop, toolbox, 0.5, 0.1, 40, alpha=0.05, beta=10, gamma=0.25, rho=0.9, stats=mstats,
    #                   halloffame=hof, verbose=True)
    # print log
    function = gp.compile(hof[0], pset)
    gen_ys = np.array([function(x, f) for x in t])
    gen_sr = target.framerate
    generated = thinkdsp.Wave(ys=gen_ys, framerate=gen_sr)
    generated.write(filename="generated.wav")
    #plot_log(log, 'aaa')
    return pop, log, hof


if __name__ == "__main__":
    main()
