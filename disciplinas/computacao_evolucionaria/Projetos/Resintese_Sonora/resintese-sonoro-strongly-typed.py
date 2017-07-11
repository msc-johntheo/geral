# Util
import operator
import math
import random
import numpy as np
from numpy import linspace, sin, pi, int16
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
import thinkplot

# Multi-processing
#import multiprocessing
#import scoop
#from scoop import futures

# DADOS INICIAIS
wave = thinkdsp.read_wave('sounds/92002__jcveliz__violin-origional.wav')
target = wave.segment(1.18995, 0.62)
f0 = target.make_spectrum().peaks()[0][1]  # frequencia fundamental
framerate = target.framerate
duration = target.duration
ts = target.ts  # array de tempo


# FUNCOES OPERADORES
def gen_sin(freq, amp=1):
    return thinkdsp.Sinusoid(freq=freq, amp=amp, func=np.sin).evaluate(target.ts).tolist()


def gen_cos(freq, amp=1, offset=0):
    signal = thinkdsp.CosSignal(freq=freq, amp=amp, offset=offset)
    return signal.make_wave(duration=duration, framerate=framerate).ys.tolist()


def signal_sum(signal1, signal2):
    return operator.add(np.array(signal1), np.array(signal2)).tolist()


def signal_mul(a, signal):
    return [x * a for x in signal]


def fm_mod(amp_carrier, freq_carrier, amp_wave, freq_wave):
    fm = amp_wave * np.sin(2 * np.pi * freq_wave * ts + amp_carrier * np.sin(2 * np.pi * freq_carrier * ts))
    return fm.tolist()


def amp(amp_float):
    return amp_float


def random_sin():
    freq = random.uniform(30, 10000)
    return gen_sin(freq)


# FUNCOES AUXILIARES
def extract_features(signal):
    X = np.array(signal, dtype=np.float32)
    sample_rate = framerate
    features = np.empty((0, 193))

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

    return features


def extract_features_separated(signal):
    X = np.array(signal, dtype=np.float32)
    sample_rate = framerate

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    f = target.make_spectrum().peaks()[0][1]  # frequencia fundamental

    return mfccs, chroma, mel, contrast, tonnetz


# PROGRAMAÇÂO GENETICA
pset = gp.PrimitiveSetTyped("MAIN", [float], list, "F")

# OPERATORS
pset.addPrimitive(gen_sin, [float, float], list)
pset.addPrimitive(signal_sum, [list, list], list)
pset.addPrimitive(fm_mod, [float, float, float, float], list)
pset.addPrimitive(amp, [float], float)
# pset.addPrimitive(gen_cos, [float,float], list)

# TERMINALS
#if not scoop.IS_ORIGIN:
#   pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addTerminal(gen_sin(f0), list)
pset.addTerminal(gen_sin(2 * f0), list)
pset.addTerminal(gen_sin(3 * f0), list)
pset.addTerminal(gen_sin(4 * f0), list)
pset.addTerminal(gen_sin(5 * f0), list)
pset.addTerminal(gen_sin(6 * f0), list)
pset.addTerminal(gen_sin(7 * f0), list)
pset.addTerminal(gen_sin(8 * f0), list)
pset.addTerminal(gen_sin(9 * f0), list)
pset.addTerminal(gen_sin(10 * f0), list)
pset.addTerminal(f0, float)
pset.addTerminal(2 * f0, float)
pset.addTerminal(3 * f0, float)
pset.addTerminal(4 * f0, float)
pset.addTerminal(5 * f0, float)
pset.addTerminal(6 * f0, float)
pset.addTerminal(7 * f0, float)
pset.addTerminal(8 * f0, float)
pset.addTerminal(9 * f0, float)
pset.addTerminal(10 * f0, float)

# CONFIG
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# toolbox.register("map", futures.map)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# FITNESS 1
def mag_pha_s_error(target, generated):
    # obtendo os parametros
    target_spectrum = target.make_spectrum()
    target_pha = target_spectrum.angles
    target_mag = np.absolute(target_spectrum.hs)
    generated_spectrum = generated.make_spectrum()
    generated_pha = generated_spectrum.angles
    generated_mag = np.absolute(generated_spectrum.hs)

    # TODO: pensar em uma forma melhor de particionar
    partitions = 11

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
    target.normalize()
    generated.normalize()

    target_peaks = np.array(target.make_spectrum().peaks())
    generated_peaks = np.array(generated.make_spectrum().peaks())
    squared_error = (target_peaks - generated_peaks) ** 2
    squared_error = squared_error / (squared_error[:, 0].max(), squared_error[:, 1].max())

    return squared_error.mean()


# FITNESS 2
def eval_mag(individual):
    signal_function = toolbox.compile(expr=individual)
    signal = signal_function(f0)
    generated = thinkdsp.Wave(ys=signal, ts=target.ts, framerate=target.framerate)

    return mag_fre_s_error(target, generated),


# FITNESS 3
def features_diff(signal1, signal2):
    diff = extract_features(signal1) - extract_features(signal2)
    return ((diff / diff.max()) ** 2).mean(axis=0)


# FITNESS 3
def eval_features(individual):
    signal_function = toolbox.compile(expr=individual)
    signal = signal_function(f0)

    return features_diff(target.ys, signal),


def eval_dummy(individual):
    signal_function = toolbox.compile(expr=individual)
    return 1,


toolbox.register("evaluate", eval_mag)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


def main():
    random.seed(10)
    pop = toolbox.population(n=3000)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.75, 0.2, 50, stats, halloffame=hof)

    signal_function = gp.compile(hof[0], pset)
    signal = signal_function(f0)
    signal_generated = thinkdsp.Wave(ys=signal, framerate=framerate)
    signal_generated.normalize()
    signal_generated.write(filename="generated.wav")
    print(str(hof[0]))
    return pop, stats, hof


if __name__ == "__main__":
    main()
