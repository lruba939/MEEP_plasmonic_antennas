import numpy as np
import matplotlib.pyplot as plt
import sys

def freqs(wav0, df):
    f0 = 1 / wav0
    fmin = f0 - df
    fmax = f0 + df
    print("\nFrequency range: ", fmin, "-", fmax, "   with center at ", f0, ".\n")    

def wavelengths(wav0, df):
    f0 = 1 / wav0
    fmin = f0 - df
    fmax = f0 + df
    wavmax = 1/fmin
    wavmin = 1/fmax
    print("\nWavelength range: ", wavmin, "-", wavmax, "   with center at ", wav0, ".\n")    

if __name__ == "__main__":
    wav0 = float(sys.argv[1])
    df = float(sys.argv[2])
    freqs(wav0, df)
    wavelengths(wav0, df)
