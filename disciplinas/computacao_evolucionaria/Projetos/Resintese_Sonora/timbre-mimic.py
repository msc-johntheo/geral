# Util
import operator
import math
import random
import numpy as np
from numpy import linspace, sin, pi, int16
import matplotlib.pyplot as plt
import scipy.io.wavfile

# Genetic Programming
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

rate, data = scipy.io.wavfile.read('sounds/92002__jcveliz__violin-origional.wav')

print(rate)